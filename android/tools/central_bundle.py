#!/usr/bin/env python3
"""Validate the Android Central publication, optionally sign it, and bundle it.

Unsigned validation never creates a ZIP. Signing uses an existing GPG key only;
this helper neither creates keys nor uploads artifacts.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile


GROUP = "io.github.quangng2000"
ARTIFACT = "reuse-audio-android"
STREAMING_ARTIFACT = "reuse-streaming-android"
ORT_VERSION = "1.23.2"
ABIS = {"arm64-v8a", "armeabi-v7a", "x86", "x86_64"}
CHECKSUMS = ("md5", "sha1", "sha256", "sha512")
LICENSES = {
    "MODEL-LICENSE.txt", "OpenMDW-1.1.txt", "POCKETFFT-LICENSE.txt",
    "ONNXRUNTIME-LICENSE.txt", "ONNXRUNTIME-THIRD-PARTY-NOTICES.txt",
    "THIRD-PARTY-NOTICES.txt",
}
API_SOURCES = {"ReuseEnhancer.kt", "ModelManifest.kt", "CancellationToken.kt"}
LICENSE_ASSETS = (Path(__file__).resolve().parents[1] / "reuse-audio" / "src" / "main"
                  / "assets" / "reuse-audio" / "licenses")
PROFILES = {
    ARTIFACT: {
        "assets": LICENSE_ASSETS, "licenses": LICENSES, "api": API_SOURCES,
        "package": b"package io.riftco.reuseaudio", "asset_prefix": "assets/reuse-audio/licenses/",
        "adapter": "reuse_audio_dsp.cpp", "library": "libreuseaudio_dsp.so",
    },
    STREAMING_ARTIFACT: {
        "assets": (Path(__file__).resolve().parents[1] / "reuse-streaming" / "src" / "main"
                   / "assets" / "reuse-streaming" / "licenses"),
        "licenses": {"LICENSE.txt", "THIRD_PARTY_NOTICES.md", "POCKETFFT_LICENSE.txt",
                     "ADAPTER_LICENSE.txt", "ONNXRUNTIME_LICENSE.txt", "ONNXRUNTIME_THIRD_PARTY_NOTICES.txt"},
        "api": {"StreamingEnhancer.kt", "StreamingEnhancerConfiguration.kt", "StreamingModelManifest.kt",
                "StreamingCancellationToken.kt", "StreamingException.kt"},
        "package": b"package io.riftco.reusestreaming", "asset_prefix": "assets/reuse-streaming/licenses/",
        "adapter": "reuse_audio_dsp.cpp", "library": "libreusestreaming_dsp.so",
    },
}
STREAMING_LICENSE = Path(__file__).resolve().parents[2] / "apple" / "Research" / "LICENSE.txt"
FORBIDDEN_SUFFIXES = {
    ".onnx", ".safetensors", ".pt", ".pth", ".ckpt", ".tflite", ".npy",
    ".npz", ".mlmodel", ".bin", ".pem", ".key", ".p12", ".pfx", ".jks",
    ".keystore", ".gpg",
}
SECRET_MARKERS = (
    b"-----BEGIN PRIVATE KEY-----", b"-----BEGIN RSA PRIVATE KEY-----",
    b"-----BEGIN EC PRIVATE KEY-----", b"-----BEGIN OPENSSH PRIVATE KEY-----",
    b"-----BEGIN PGP PRIVATE KEY BLOCK-----",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data, algorithm):
    return hashlib.new(algorithm, data).hexdigest()


def safe_content(name, data, artifact_id=ARTIFACT):
    require(artifact_id in PROFILES, "Unapproved publication artifact")
    path = PurePosixPath(name)
    lower = name.lower()
    require(not path.is_absolute() and ".." not in path.parts and "\\" not in name,
            "Unsafe archive path")
    if artifact_id == ARTIFACT:
        require(not any(term in lower for term in ("streaming", "nscl", "real-time_re-use")),
                "Streaming material is not allowed in the offline publication")
    require(not any(part.lower().endswith(".mlmodelc") for part in path.parts),
            "Model weights are not allowed in the publication")
    require(path.suffix.lower() not in FORBIDDEN_SUFFIXES,
            "Model weights or private key files are not allowed")
    require(not any(part.lower() in {".git", ".ssh", ".gnupg", ".netrc", ".npmrc", ".pypirc",
                                    "id_rsa", "id_ed25519", "local.properties", "gradle.properties",
                                    "signing.properties", "key.properties", "token.txt"}
                    or part.lower().startswith((".env", "credentials", "secrets", "password", "passphrase"))
                    for part in path.parts),
            "A private configuration file is present")
    require(not any(marker in data for marker in SECRET_MARKERS),
            "Private key material is present")
    require(not re.search(rb"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[A-Z0-9]{16})", data),
            "A credential-like token is present")


def zip_contents(path, artifact_id=ARTIFACT):
    """Read bounded ZIP contents, including the AAR's nested classes.jar."""
    budget = [256 * 1024 * 1024]

    def inspect(source, depth=0):
        entries = {}
        with zipfile.ZipFile(source) as archive:
            for entry in archive.infolist():
                require(entry.filename not in entries, "Duplicate archive entry")
                require((entry.external_attr >> 16) & 0o170000 != 0o120000,
                        "Archive symlinks are not allowed")
                require(entry.file_size <= 64 * 1024 * 1024,
                        "Unexpectedly large archive member")
                budget[0] -= entry.file_size
                require(budget[0] >= 0, "Archive exceeds the validation size limit")
                data = archive.read(entry)
                safe_content(entry.filename, data, artifact_id)
                entries[entry.filename] = data
                if entry.filename.endswith(".jar") and not entry.is_dir():
                    require(depth < 2, "Unexpected nested archive depth")
                    inspect(io.BytesIO(data), depth + 1)
        return entries

    return inspect(path)


def verify_licenses(entries, artifact_id=ARTIFACT):
    profile = PROFILES[artifact_id]
    names = {PurePosixPath(name).name for name in entries}
    require(profile["licenses"] <= names, "Artifact is missing required license notices")
    for source in profile["assets"].iterdir():
        if source.is_file():
            copies = [data for name, data in entries.items() if PurePosixPath(name).name == source.name]
            require(copies and all(data == source.read_bytes() for data in copies),
                    "Artifact license notice differs from the checked-in source")
    if artifact_id == STREAMING_ARTIFACT:
        require(any(PurePosixPath(name).name == "LICENSE.txt" and data == STREAMING_LICENSE.read_bytes()
                    for name, data in entries.items()), "Streaming artifact must retain the exact NSCLv1 license")
    else:
        require(not any(b"3.3 Use Limitation." in data for data in entries.values()),
                "Offline artifact must not contain the streaming license")
    require(any(b"quangng2000" in data and b"Permission is hereby granted" in data
                and b"MIT" in data for name, data in entries.items()
                if "license" in name.lower() or "notice" in name.lower()),
            "Artifact is missing the adapter MIT license and attribution")


def verify_pom(path, version, artifact_id=ARTIFACT):
    data = path.read_bytes()
    safe_content(path.name, data, artifact_id)
    pom = ET.fromstring(data)
    namespace = {"m": "http://maven.apache.org/POM/4.0.0"}

    def value(node, field):
        return (node.findtext("m:" + field.replace("/", "/m:"), "", namespace)).strip()

    require(pom.tag == "{http://maven.apache.org/POM/4.0.0}project", "Invalid POM namespace")
    for field, expected in {"modelVersion": "4.0.0", "groupId": GROUP,
                            "artifactId": artifact_id, "version": version, "packaging": "aar"}.items():
        require(value(pom, field) == expected, "Incorrect POM " + field)
    for field in ("name", "description", "url", "scm/connection", "scm/developerConnection", "scm/url"):
        require(value(pom, field), "POM is missing " + field)
    require(value(pom, "url").startswith("https://"), "POM project URL must use HTTPS")
    licenses = pom.findall("m:licenses/m:license", namespace)
    require(licenses and all(value(item, "name") and value(item, "url") for item in licenses),
            "POM license names and URLs are required")
    license_names = [value(item, "name") for item in licenses]

    def is_streaming_license(name):
        return "NSCL" in name or "NVIDIA One-Way Noncommercial" in name

    if artifact_id == STREAMING_ARTIFACT:
        require(any(is_streaming_license(name) for name in license_names),
                "POM must identify the streaming NSCLv1 license")
        require(not any("OpenMDW" in name for name in license_names),
                "Streaming POM must not claim the offline model license")
    else:
        require(any("OpenMDW" in name for name in license_names),
                "POM must identify the offline OpenMDW license")
        require(not any(is_streaming_license(name) for name in license_names),
                "POM must not apply the streaming license")
    developers = pom.findall("m:developers/m:developer", namespace)
    require(developers and all(value(item, "name") and (value(item, "url") or value(item, "email"))
                               for item in developers), "POM developer identity and contact are required")
    dependencies = pom.findall("m:dependencies/m:dependency", namespace)
    require(not any(value(item, "groupId") == GROUP and value(item, "artifactId") in PROFILES
                    for item in dependencies), "Offline and research artifacts must remain independent")
    ort = [item for item in dependencies if value(item, "groupId") == "com.microsoft.onnxruntime"
           and value(item, "artifactId") == "onnxruntime-android"]
    require(len(ort) == 1 and value(ort[0], "version") == ORT_VERSION
            and value(ort[0], "scope") == "runtime" and value(ort[0], "optional") != "true",
            "POM must retain the pinned ONNX Runtime Android runtime dependency")
    require(not pom.findall("m:repositories", namespace), "POM must resolve dependencies from Central")


def verify_archives(aar, sources, documentation, artifact_id=ARTIFACT):
    profile = PROFILES[artifact_id]
    entries = zip_contents(aar, artifact_id)
    verify_licenses(entries, artifact_id)
    require(all(name.endswith("/") or name.startswith(profile["asset_prefix"])
                for name in entries if name.startswith("assets/")),
            "AAR assets must contain license notices only")
    require("classes.jar" in entries and len(entries["classes.jar"]) > 100,
            "AAR is missing compiled classes")
    native = {name for name in entries if name.startswith("jni/") and not name.endswith("/")}
    require(native == {f"jni/{abi}/{profile['library']}" for abi in ABIS},
            "AAR must contain the DSP library for exactly the four supported ABIs")
    require(all(entries[name].startswith(b"\x7fELF") for name in native), "Invalid JNI binary")
    source_entries = zip_contents(sources, artifact_id)
    verify_licenses(source_entries, artifact_id)
    source_names = {PurePosixPath(name).name for name, data in source_entries.items() if len(data) > 100}
    require(profile["api"] <= source_names and {profile["adapter"], "pocketfft_hdronly.h"} <= source_names,
            "Sources JAR must include the Kotlin API and native DSP sources")
    for filename in profile["api"]:
        require(any(PurePosixPath(name).name == filename and profile["package"] in data
                    for name, data in source_entries.items()), "Sources JAR contains invalid Kotlin API sources")
    doc_entries = zip_contents(documentation, artifact_id)
    verify_licenses(doc_entries, artifact_id)
    require(any(PurePosixPath(name).name == "README.md" and len(data.strip()) >= 100
                for name, data in doc_entries.items()), "Documentation JAR requires a useful README.md")
    documented = {PurePosixPath(name).name for name, data in doc_entries.items()
                  if name.startswith("api/") and len(data) > 100 and b"/**" in data}
    require(profile["api"] <= documented, "Documentation JAR must include the public Kotlin API KDoc")


def verify_module(path, primary, version, artifact_id=ARTIFACT):
    data = path.read_bytes()
    safe_content(path.name, data, artifact_id)
    module = json.loads(data)
    component = module.get("component", {})
    require((component.get("group"), component.get("module"), component.get("version"))
            == (GROUP, artifact_id, version), "Incorrect Gradle module coordinates")
    for variant in module.get("variants", []):
        for item in variant.get("files", []):
            name = item.get("url")
            require(name in primary and name != path.name, "Gradle module references an unknown artifact")
            contents = primary[name].read_bytes()
            require(item.get("size") == len(contents), "Gradle module artifact size mismatch")
            for algorithm in CHECKSUMS:
                if algorithm in item:
                    require(item[algorithm] == digest(contents, algorithm), "Gradle module checksum mismatch")


def gpg_command(gpg_home):
    require(shutil.which("gpg"), "GPG is required to sign or verify publication artifacts")
    command = ["gpg", "--batch", "--no-tty", "--no-auto-key-retrieve"]
    if gpg_home:
        command.extend(["--homedir", str(gpg_home.expanduser().resolve())])
    return command


def verify_signature(command, artifact, artifact_id=ARTIFACT):
    signature = Path(str(artifact) + ".asc").read_bytes()
    safe_content(artifact.name + ".asc", signature, artifact_id)
    require(signature.lstrip().startswith(b"-----BEGIN PGP SIGNATURE-----"),
            "Publication signatures must be ASCII-armored")
    result = subprocess.run(command + ["--status-fd", "1", "--verify", str(artifact) + ".asc", str(artifact)],
                            stdin=subprocess.DEVNULL, capture_output=True, check=False, timeout=30)
    require(result.returncode == 0 and b"[GNUPG:] VALIDSIG " in result.stdout,
            "GPG signature verification failed for " + artifact.name)


def collect_publication(repository, version, artifact_id, require_module=False):
    relative = Path(*GROUP.split("."), artifact_id, version)
    release = repository / relative
    require(release.is_dir(), "Publication version directory does not exist")
    require(release.resolve() == release, "Publication directories must not contain symlinks")
    base = f"{artifact_id}-{version}"
    names = [base + suffix for suffix in (".aar", ".pom", "-sources.jar", "-javadoc.jar")]
    require(all((release / name).is_file() for name in names), "Required publication artifact is missing")
    if (release / (base + ".module")).exists():
        names.append(base + ".module")
    require(not require_module or base + ".module" in names,
            "Cohort publications require Gradle module metadata for both artifacts")
    primary = {name: release / name for name in names}
    allowed = {name + suffix for name in names
               for suffix in ("", ".asc") + tuple("." + algorithm for algorithm in CHECKSUMS)
               + tuple(".asc." + algorithm for algorithm in CHECKSUMS)}
    for path in release.iterdir():
        require(not path.is_symlink() and path.is_file(), "Publication must contain ordinary files only")
        require(path.name in allowed or path.name.startswith("maven-metadata.xml"),
                "Unexpected file in publication version directory")
    verify_pom(primary[base + ".pom"], version, artifact_id)
    verify_archives(primary[base + ".aar"], primary[base + "-sources.jar"], primary[base + "-javadoc.jar"], artifact_id)
    if base + ".module" in primary:
        verify_module(primary[base + ".module"], primary, version, artifact_id)
    return primary


def bundle(args):
    require(re.fullmatch(r"[0-9][A-Za-z0-9_.+-]*", args.version)
            and not args.version.endswith("-SNAPSHOT"), "A release version is required")
    repository = args.repository.expanduser().resolve()
    include_streaming = getattr(args, "include_streaming", False)
    artifact_ids = [ARTIFACT, STREAMING_ARTIFACT] if include_streaming else [ARTIFACT]
    primary, owners = {}, {}
    # Validate the complete cohort before inspecting keys or signing any artifact.
    for artifact_id in artifact_ids:
        artifacts = collect_publication(repository, args.version, artifact_id, require_module=include_streaming)
        primary.update(artifacts)
        owners.update({name: artifact_id for name in artifacts})
    passphrase_file = getattr(args, "passphrase_file", None)
    require(not passphrase_file or args.signing_key, "--passphrase-file requires --signing-key")
    if args.signing_key:
        command = gpg_command(args.gpg_home)
        if passphrase_file:
            passphrase_file = passphrase_file.expanduser().resolve()
            require(passphrase_file.is_file(), "Passphrase file does not exist")
            require(repository != passphrase_file and repository not in passphrase_file.parents,
                    "Passphrase file must be outside the staging repository")
            command.extend(["--pinentry-mode", "loopback", "--passphrase-file", str(passphrase_file)])
        for artifact in primary.values():
            result = subprocess.run(command + ["--yes", "--local-user", args.signing_key, "--armor",
                                    "--detach-sign", "--output", str(artifact) + ".asc", str(artifact)],
                                    stdin=subprocess.DEVNULL, capture_output=True, check=False, timeout=60)
            require(result.returncode == 0, "GPG signing failed for " + artifact.name)
    signed = [artifact for artifact in primary.values() if Path(str(artifact) + ".asc").is_file()]
    if signed:
        command = gpg_command(args.gpg_home)
        for artifact in signed:
            verify_signature(command, artifact, owners[artifact.name])
    ready = len(signed) == len(primary)
    require(ready or args.allow_unsigned, "Missing signatures; use --allow-unsigned for validation without a ZIP")
    coordinates = [f"{GROUP}:{artifact_id}:{args.version}" for artifact_id in artifact_ids]
    report = {"schemaVersion": 2, "artifactIds": artifact_ids, "coordinateSet": coordinates,
              "coordinates": coordinates if include_streaming else coordinates[0], "uploadReady": ready,
              "artifactCount": len(primary), "verifiedSignatureCount": len(signed),
              "artifacts": [{"name": name, "bytes": path.stat().st_size,
                             "sha256": digest(path.read_bytes(), "sha256")}
                            for name, path in sorted(primary.items())], "bundleCreated": False}
    if not ready:
        return report
    output = args.output.expanduser().resolve()
    require(repository not in output.parents, "Output ZIP must be outside the staging repository")
    output.parent.mkdir(parents=True, exist_ok=True)
    files = []
    for name, path in primary.items():
        files.extend([path, Path(str(path) + ".asc")])
        data = path.read_bytes()
        for algorithm in CHECKSUMS:
            sidecar = Path(str(path) + "." + algorithm)
            sidecar.write_text(digest(data, algorithm) + "\n", encoding="ascii")
            files.append(sidecar)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=output.name + ".", dir=output.parent, delete=False) as handle:
            temporary = Path(handle.name)
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(files, key=lambda item: item.relative_to(repository).as_posix()):
                info = zipfile.ZipInfo(path.relative_to(repository).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, path.read_bytes())
        temporary.replace(output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    report.update(bundleCreated=True, entryCount=len(files), bundleBytes=output.stat().st_size,
                  bundleSha256=digest(output.read_bytes(), "sha256"))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--version", default="0.1.2")
    parser.add_argument("--include-streaming", action="store_true",
                        help="Validate and publish the separate NSCLv1 research artifact in the same bundle")
    parser.add_argument("--gpg-home", type=Path)
    parser.add_argument("--signing-key")
    parser.add_argument("--passphrase-file", type=Path,
                        help="Read an existing signing key's passphrase from a file outside the repository")
    parser.add_argument("--allow-unsigned", action="store_true",
                        help="Validate locally; never create a ZIP with missing signatures")
    args = parser.parse_args()
    try:
        print(json.dumps(bundle(args), indent=2, sort_keys=True))
    except subprocess.TimeoutExpired:
        print(json.dumps({"ok": False, "error": "GPG operation timed out"}), file=sys.stderr)
        return 1
    except (ValueError, OSError, ET.ParseError, zipfile.BadZipFile) as error:
        # Avoid subprocess output, key identities, source contents, and credentials.
        print(json.dumps({"ok": False, "error": str(error)}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
