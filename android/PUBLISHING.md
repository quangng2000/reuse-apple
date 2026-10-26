# Android Maven Central publishing

Coordinate: `io.github.quangng2000:reuse-audio-android:0.1.1`.
The Android artifact version is independent of the repository-wide Swift release
version. The matching source tag is `android-v0.1.1`.

The release AAR contains all four native ABIs and license notices. Model weights
remain a separate, hash-verified download. ONNX Runtime is a transitive runtime
dependency in the published POM and Gradle module metadata.

## Prepare the release

Use the pinned Android toolchain in [README.md](README.md). From `android/`:

```sh
./gradlew :reuse-audio:publishReleasePublicationToCentralStagingRepository --max-workers=2
python3 tools/central_bundle.py \
  --repository reuse-audio/build/central-repository \
  --output reuse-audio/build/reuse-audio-android-0.1.1-central.zip \
  --version 0.1.1 --allow-unsigned
```

The unsigned check validates artifacts without creating an upload ZIP when
signatures are missing. The local repository is under the ignored build folder.
Source JARs contain Kotlin, native adapter/PocketFFT sources and notices. The
Javadoc-classifier JAR contains the integration guide, API KDoc in readable Kotlin
source, and licenses; it is not generated HTML Javadoc.

## Sign and bundle

Maven Central requires a verified namespace and public-key signatures. Keep the
private signing key and its passphrase outside this repository, with owner-only
permissions, and back them up securely. No portal token is needed for a manual
browser upload. Never add secrets, private keys or passphrases to a commit.

```sh
python3 tools/central_bundle.py \
  --repository reuse-audio/build/central-repository \
  --output reuse-audio/build/reuse-audio-android-0.1.1-central.zip \
  --version 0.1.1 \
  --gpg-home /absolute/private/gnupg \
  --signing-key YOUR_PUBLIC_FINGERPRINT \
  --passphrase-file /absolute/private/signing-passphrase
```

The helper verifies the POM, sources, documentation, native ABIs, notices and
signatures. It creates MD5/SHA-1/SHA-256/SHA-512 checksums and a ZIP using the exact
Maven repository directory layout. No model weights, metadata indexes or secret
files are included. It does not upload or publish anything.

Upload the signed ZIP at [Central Publishing](https://central.sonatype.com/publishing).
Wait for validation, inspect the component coordinate and publish it. Releases on
Maven Central are immutable: never overwrite an existing version. Verify the
public POM and AAR at `https://repo.maven.apache.org/maven2/` before describing a
release as available. The project tag must exist before the corresponding
package is published.

References: [Sonatype requirements](https://central.sonatype.org/publish/requirements/),
[manual bundle upload](https://central.sonatype.org/publish/publish-portal-upload/),
[namespace ownership](https://central.sonatype.org/register/namespace/).

## Current release signing key

Public fingerprint: `93C571756E6EF6906F88DFD2A37754DCEBAECBBE`.
The [public key](release-signing-key.asc) is also available from
`hkps://keyserver.ubuntu.com`. Its identity uses the maintainer's public GitHub
noreply address. The initial encrypted private key, revocation certificate and
passphrase are kept outside the checkout under
`~/.config/reuse-audio/publishing/`; back up that directory securely for future
releases. Public material may be redistributed; private material must never be
uploaded or committed.
