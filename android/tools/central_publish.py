#!/usr/bin/env python3
"""Publish an already signed Android bundle to Central and verify public bytes.

Credentials are read only from CENTRAL_TOKEN_USERNAME/CENTRAL_TOKEN_PASSWORD.
Run central_bundle.py first to validate/sign the actual build. This client checks
the upload's layout, coordinates, checksums and signatures before any request.
Mutating requests are never retried. A matching report with a known deployment
ID can be recovered using --deployment-id. Unknown upload outcomes require
review in the Central portal; this client never guesses a deployment.
"""

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import xml.etree.ElementTree as ET
import zipfile


GROUP = "io.github.quangng2000"
ARTIFACT = "reuse-audio-android"
STREAMING_ARTIFACT = "reuse-streaming-android"
ARTIFACT_IDS = (ARTIFACT, STREAMING_ARTIFACT)
API_ORIGIN = "https://central.sonatype.com"
PUBLIC_ORIGIN = "https://repo.maven.apache.org"
CHECKSUMS = ("md5", "sha1", "sha256", "sha512")
STATES = {"PENDING", "VALIDATING", "VALIDATED", "PUBLISHING", "PUBLISHED", "FAILED"}
MAX_RESPONSE = 64 * 1024 * 1024


class PublishError(Exception):
    """A safe, fixed diagnostic code; never include server bodies or credentials."""


def require(condition, code):
    if not condition:
        raise PublishError(code)


def sha256(contents):
    return hashlib.sha256(contents).hexdigest()


def validate_version(version):
    require(re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:-[A-Za-z0-9][A-Za-z0-9.-]*)?", version)
            and not version.endswith("-SNAPSHOT"), "invalid_release_version")


def validate_deployment_id(value):
    try:
        require(str(uuid.UUID(value)) == value, "invalid_deployment_id")
    except (ValueError, AttributeError, TypeError):
        raise PublishError("invalid_deployment_id") from None
    return value


def load_bundle(path, version, include_streaming=False):
    """Validate exact Maven paths and checksum coverage without extracting files."""
    validate_version(version)
    require(path.is_file() and not path.is_symlink(), "bundle_missing_or_not_regular")
    require(0 < path.stat().st_size <= MAX_RESPONSE, "bundle_size_limit")
    contents = path.read_bytes()
    artifact_ids = ARTIFACT_IDS if include_streaming else (ARTIFACT,)
    components = [{"artifactId": artifact_id, "coordinate": GROUP + ":" + artifact_id + ":" + version,
                   "prefix": GROUP.replace(".", "/") + "/" + artifact_id + "/" + version + "/"}
                  for artifact_id in artifact_ids]
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            require(len(entries) <= 30 * len(artifact_ids) and len({e.filename for e in entries}) == len(entries),
                    "bundle_duplicate_or_extra_entries")
            require(all(any(e.filename.startswith(component["prefix"]) for component in components) and not e.is_dir()
                        and not (e.external_attr >> 16) & 0o170000 == 0o120000
                        and e.file_size <= MAX_RESPONSE for e in entries), "invalid_bundle_path")
            require(sum(e.file_size for e in entries) <= MAX_RESPONSE, "bundle_expansion_limit")
            files = {e.filename: archive.read(e) for e in entries}
    except (zipfile.BadZipFile, RuntimeError, OSError):
        raise PublishError("invalid_bundle_archive") from None
    primary = set()
    for component in components:
        base = component["artifactId"] + "-" + version
        prefix = component["prefix"]
        primary.update(prefix + base + suffix for suffix in (".aar", ".pom", "-sources.jar", "-javadoc.jar"))
        if include_streaming or prefix + base + ".module" in files:
            primary.add(prefix + base + ".module")
    expected = {name + suffix for name in primary
                for suffix in ("", ".asc") + tuple("." + algorithm for algorithm in CHECKSUMS)}
    require(set(files) == expected, "incomplete_or_unexpected_bundle_files")
    for name in primary:
        require(files[name], "empty_publication_artifact")
        for algorithm in CHECKSUMS:
            expected_hash = hashlib.new(algorithm, files[name]).hexdigest().encode("ascii")
            require(files[name + "." + algorithm].strip() == expected_hash, "bundle_checksum_mismatch")
        signature = files[name + ".asc"].strip()
        require(signature.startswith(b"-----BEGIN PGP SIGNATURE-----")
                and signature.endswith(b"-----END PGP SIGNATURE-----"), "missing_armored_signature")
    ns = "{http://maven.apache.org/POM/4.0.0}"
    for component in components:
        artifact_id = component["artifactId"]
        try:
            pom = ET.fromstring(files[component["prefix"] + artifact_id + "-" + version + ".pom"])
        except ET.ParseError:
            raise PublishError("invalid_bundle_pom") from None
        require(pom.tag == ns + "project", "invalid_bundle_pom")
        for key, expected_value in {"groupId": GROUP, "artifactId": artifact_id,
                                    "version": version, "packaging": "aar"}.items():
            require(pom.findtext(ns + key) == expected_value, "bundle_coordinate_mismatch")
    # Basenames remain unique across these two explicit artifact IDs. Keep the
    # legacy singleton shape while recording each full repository path.
    return {"bytes": contents, "sha256": sha256(contents), "prefix": components[0]["prefix"],
            "components": components, "coordinates": [item["coordinate"] for item in components],
            "paths": {name.rsplit("/", 1)[1]: name for name in sorted(primary)},
            "primary": {name.rsplit("/", 1)[1]: files[name] for name in sorted(primary)}}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        raise PublishError("http_redirect_refused")


class CentralClient:
    def __init__(self, username, password, deadline, clock=time.monotonic, opener=None):
        require(username and password and ":" not in username, "missing_central_credentials")
        require(not any(c in username + password for c in "\r\n"), "invalid_central_credentials")
        self.authorization = "Bearer " + base64.b64encode((username + ":" + password).encode()).decode()
        self.deadline, self.clock = deadline, clock
        self.opener = opener or urllib.request.build_opener(NoRedirect())

    def _request(self, url, method="GET", data=None, headers=None, authenticated=False,
                 missing_ok=False, operation="request"):
        parts = urllib.parse.urlsplit(url)
        expected_host = "central.sonatype.com" if authenticated else "repo.maven.apache.org"
        require(parts.scheme == "https" and parts.netloc == expected_host
                and not parts.fragment, "request_destination_refused")
        remaining = self.deadline - self.clock()
        require(remaining > 0, "publication_timeout")
        request_headers = {"User-Agent": "reuse-audio-central-publisher/1", **(headers or {})}
        require(not any(k.lower() == "authorization" for k in request_headers), "authorization_override_refused")
        if authenticated:
            request_headers["Authorization"] = self.authorization
        request = urllib.request.Request(url, data=data, headers=request_headers, method=method)
        try:
            with self.opener.open(request, timeout=min(60, remaining)) as response:
                allowed_statuses = {"upload": {200, 201}, "publish": {200, 204}}
                require(response.status in allowed_statuses.get(operation, {200}),
                        operation + "_unexpected_status")
                result = response.read(MAX_RESPONSE + 1)
                require(len(result) <= MAX_RESPONSE, "response_size_limit")
                return result
        except urllib.error.HTTPError as error:
            if error.code == 404 and missing_ok:
                return None
            # Response bodies, reason strings and redirect URLs can contain secrets.
            raise PublishError(operation + "_http_" + str(int(error.code))) from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise PublishError(operation + "_network_error") from None

    def public_file(self, relative):
        parts = relative.split("/")
        require(len(parts) == 6 and parts[:3] == GROUP.split(".") and parts[3] in ARTIFACT_IDS,
                "invalid_public_artifact_path")
        validate_version(parts[4])
        require(parts[5] in {parts[3] + "-" + parts[4] + suffix
                            for suffix in (".aar", ".pom", ".module", "-sources.jar", "-javadoc.jar")},
                "invalid_public_artifact_path")
        return self._request(PUBLIC_ORIGIN + "/maven2/" + relative, missing_ok=True, operation="public_read")

    def upload(self, bundle, version):
        boundary = "reuse-audio-" + secrets.token_hex(16)
        filename = ARTIFACT + "-" + version + "-central.zip"
        body = ("--" + boundary + '\r\nContent-Disposition: form-data; name="bundle"; filename="'
                + filename + '"\r\nContent-Type: application/octet-stream\r\n\r\n').encode()
        body += bundle["bytes"] + ("\r\n--" + boundary + "--\r\n").encode()
        query = urllib.parse.urlencode({"name": ARTIFACT + "-" + version, "publishingType": "USER_MANAGED"})
        result = self._request(API_ORIGIN + "/api/v1/publisher/upload?" + query, "POST", body,
                               {"Content-Type": "multipart/form-data; boundary=" + boundary},
                               authenticated=True, operation="upload")
        try:
            return validate_deployment_id(result.decode("ascii").strip())
        except UnicodeDecodeError:
            raise PublishError("invalid_upload_response") from None

    def status(self, deployment_id):
        validate_deployment_id(deployment_id)
        raw = self._request(API_ORIGIN + "/api/v1/publisher/status?"
                            + urllib.parse.urlencode({"id": deployment_id}), "POST", b"",
                            authenticated=True, operation="status")
        try:
            result = json.loads(raw)
            require(isinstance(result, dict), "invalid_status_response")
            return result
        except (ValueError, UnicodeDecodeError):
            raise PublishError("invalid_status_response") from None

    def publish(self, deployment_id):
        validate_deployment_id(deployment_id)
        return self._request(API_ORIGIN + "/api/v1/publisher/deployment/" + deployment_id,
                              "POST", b"", authenticated=True, operation="publish")


def checked_status(status, deployment_id, version, artifact_ids=(ARTIFACT,)):
    require(status.get("deploymentId") == deployment_id, "deployment_identity_mismatch")
    require(status.get("deploymentName") == ARTIFACT + "-" + version, "deployment_name_mismatch")
    state = status.get("deploymentState")
    require(state in STATES, "invalid_deployment_state")
    expected = {"pkg:maven/" + GROUP + "/" + artifact_id + "@" + version for artifact_id in artifact_ids}
    purls = status.get("purls", [])
    require(isinstance(purls, list), "invalid_deployment_coordinates")
    # Central has returned PUBLISHED with an empty purls list. That read-only
    # state still requires exact receipt ID/name and every public artifact hash
    # before success. Keep coordinates mandatory before any publish mutation.
    if state in {"VALIDATED", "PUBLISHING"}:
        require(purls, "missing_deployment_coordinates")
    normalized = set()
    for purl in purls:
        require(isinstance(purl, str), "deployment_coordinate_mismatch")
        canonical = purl[:-9] if purl.endswith("?type=aar") else purl
        require(canonical in expected,
                "deployment_coordinate_mismatch")
        normalized.add(canonical)
    if state in {"VALIDATED", "PUBLISHING"} or (state == "PUBLISHED" and purls):
        require(normalized == expected, "deployment_coordinate_mismatch")
    return state


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    report["updatedAt"] = datetime.now(timezone.utc).isoformat()
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", delete=False) as output:
        temporary = Path(output.name)
        json.dump(report, output, indent=2, sort_keys=True)
        output.write("\n")
    temporary.replace(path)


def publish_bundle(bundle, version, report_path, client, deadline, deployment_id=None,
                   clock=time.monotonic, sleep=time.sleep, interval=15):
    """One explicit publication attempt, with a durable receipt before mutations."""
    coordinate = GROUP + ":" + ARTIFACT + ":" + version
    components = bundle["components"]
    coordinates = bundle["coordinates"]
    artifact_ids = tuple(item["artifactId"] for item in components)
    if deployment_id:
        validate_deployment_id(deployment_id)
        require(report_path.is_file(), "recovery_requires_matching_receipt")
    report = {"schemaVersion": 2, "coordinate": coordinate, "coordinates": coordinates,
              "bundleSha256": bundle["sha256"],
              "publishingType": "USER_MANAGED", "published": False, "uploadRequested": False,
              "publishRequested": False, "publishAccepted": False, "states": [], "artifacts": []}
    if report_path.exists():
        try:
            prior = json.loads(report_path.read_text())
        except (ValueError, OSError):
            raise PublishError("prior_report_unreadable") from None
        require(prior.get("coordinate") == coordinate and prior.get("bundleSha256") == bundle["sha256"],
                "prior_report_artifact_mismatch")
        require(prior.get("coordinates", [prior.get("coordinate")]) == coordinates,
                "prior_report_cohort_mismatch")
        require(deployment_id, "prior_attempt_requires_explicit_recovery")
        require(prior.get("deploymentId"), "unknown_upload_outcome_requires_portal_review")
        require(prior.get("deploymentId") == deployment_id, "deployment_receipt_mismatch")
        # Copy only our bounded public state, not arbitrary fields from a report.
        previous_states = prior.get("states", [])
        require(isinstance(previous_states, list)
                and all(isinstance(state, str) and state in STATES for state in previous_states),
                "prior_report_state_invalid")
        report.update(uploadRequested=bool(prior.get("uploadRequested")),
                      publishRequested=bool(prior.get("publishRequested")),
                      publishAccepted=bool(prior.get("publishAccepted")), states=previous_states)
    try:
        for component in components:
            pom_name = component["artifactId"] + "-" + version + ".pom"
            existing = client.public_file(component["prefix"] + pom_name)
            require(deployment_id or existing is None, "version_already_published")
        if not deployment_id:
            report.update(status="UPLOAD_REQUESTED", uploadRequested=True)
            write_report(report_path, report)
            # Do not retry this request, including after a timeout/5xx response.
            deployment_id = client.upload(bundle, version)
            report.update(deploymentId=deployment_id, status="UPLOADED")
            write_report(report_path, report)
        else:
            report.update(deploymentId=deployment_id, status="RECOVERING")
            write_report(report_path, report)
        while True:
            require(clock() < deadline, "publication_timeout")
            state = checked_status(client.status(deployment_id), deployment_id, version, artifact_ids)
            report["status"] = state
            if not report["states"] or report["states"][-1] != state:
                report["states"].append(state)
            write_report(report_path, report)
            require(state != "FAILED", "central_validation_or_publication_failed")
            if state == "VALIDATED":
                if report["publishRequested"]:
                    require(report.get("publishAccepted"), "publish_outcome_requires_portal_review")
                    # A successful request may take time to appear in status reads.
                else:
                    report.update(publishRequested=True, status="PUBLISH_REQUESTED")
                    write_report(report_path, report)
                    # Do not retry this request when its response is uncertain.
                    client.publish(deployment_id)
                    report.update(publishAccepted=True)
                    write_report(report_path, report)
            if state == "PUBLISHED":
                break
            sleep(min(interval, max(0, deadline - clock())))
        # Public repository sync may lag the portal. Retry only missing reads.
        for name, expected_bytes in bundle["primary"].items():
            while True:
                require(clock() < deadline, "public_sync_timeout")
                actual = client.public_file(bundle["paths"][name])
                if actual is not None:
                    require(sha256(actual) == sha256(expected_bytes), "public_artifact_hash_mismatch")
                    report["artifacts"] = [item for item in report["artifacts"] if item["name"] != name]
                    report["artifacts"].append({"name": name, "bytes": len(actual), "sha256": sha256(actual)})
                    write_report(report_path, report)
                    break
                sleep(min(interval, max(0, deadline - clock())))
        report.update(status="PUBLIC_ARTIFACTS_VERIFIED", published=True)
        write_report(report_path, report)
        return report
    except PublishError as error:
        report["error"] = str(error)
        write_report(report_path, report)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--include-streaming", action="store_true",
                        help="Publish both the offline and separately licensed streaming research artifacts")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--deployment-id", help="Recover this ID using its matching existing report; never upload again")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args(argv)
    try:
        require(60 <= args.timeout_seconds <= 3600, "timeout_out_of_bounds")
        require(args.report.resolve() != args.bundle.resolve(), "report_must_not_overwrite_bundle")
        deadline = time.monotonic() + args.timeout_seconds
        bundle = load_bundle(args.bundle, args.version, include_streaming=args.include_streaming)
        client = CentralClient(os.environ.get("CENTRAL_TOKEN_USERNAME", ""),
                               os.environ.get("CENTRAL_TOKEN_PASSWORD", ""), deadline)
        report = publish_bundle(bundle, args.version, args.report, client, deadline,
                                deployment_id=args.deployment_id)
        print(json.dumps({"coordinate": report["coordinate"], "deploymentId": report["deploymentId"],
                          "coordinates": report["coordinates"],
                          "status": report["status"], "published": report["published"]}, sort_keys=True))
        return 0
    except PublishError as error:
        print(json.dumps({"ok": False, "error": str(error)}, sort_keys=True), file=sys.stderr)
    except Exception:
        # Avoid tracebacks containing responses, environment values or URLs.
        print(json.dumps({"ok": False, "error": "unexpected_local_error"}), file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
