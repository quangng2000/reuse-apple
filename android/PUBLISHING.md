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

## Automatic GitHub Actions releases

[Android SDK release](../.github/workflows/android-release.yml) validates Android
changes on pull requests and supports a manual validation-only run. It builds
all four native ABIs on the standard ARM64 macOS runner and runs the complete host suite with the pinned public
model and hash-verified synthetic regression fixtures. No model or numerical
behavior is changed by the workflow.

The FP32 PCM reference fixtures were validated on ARM64. The workflow uses
Apple Clang on ARM64 for those exact regression checks; Linux x86 FFT rounding
can choose a different phase branch around pi. Test limits remain unchanged.

An initial push of an `android-v<VERSION>` tag on a commit already in the default
branch triggers the isolated signing/publication job. The tag must exactly
match the literal Android `version` in `android/reuse-audio/build.gradle.kts`.
Apple/repository-wide tags such as `v0.3.1` do not publish an Android artifact.

For the next Android release:

1. Set the new Android version, for example `0.1.2`, and merge the tested change
   into `main`.
2. Create and push `android-v0.1.2` on that commit.
3. Actions tests the library, validates the staged Maven artifacts, signs them,
   uploads to Central, waits for validation, publishes once, and verifies every
   public primary artifact against the signed bundle.

One-time repository Actions secrets are required:

- `CENTRAL_TOKEN_USERNAME`: Central user-token username (not the login username).
- `CENTRAL_TOKEN_PASSWORD`: matching Central user-token password.
- `ANDROID_SIGNING_KEY`: encrypted ASCII-armored dedicated private signing key.
- `ANDROID_SIGNING_PASSPHRASE`: its passphrase.

Configure secrets only in `quangng2000/reuse-audio` repository Actions settings.
The signing job compares the imported private key's public fingerprint with the
committed `android/release-signing-key.asc`. Key rotation requires updating that
public key. Signing uses a separate runner after validation; no Gradle process
runs on it. Secret files stay under `RUNNER_TEMP`, are never artifact-uploaded,
and are removed when the step exits. Publishing secrets are not exposed to PRs,
forks, manual validation runs or reruns.

Central requires an account user token for its publishing API. Generate it in
the same Central account that owns `io.github.quangng2000`, using a descriptive
CI name and an expiration date. Do not paste credentials into issues, commits,
workflow logs or chat.

### Recover an interrupted publication

The workflow intentionally publishes only on the first attempt of a tag event.
A rerun validates again without uploading. Read the saved `publish-report.json`
in the Actions publication artifact before recovery: an upload request with an
uncertain outcome must not be blindly repeated. Resume a known deployment with
the exact original bundle, version and deployment ID:

```sh
python3 android/tools/central_publish.py \
  --bundle /absolute/path/to/the-original-signed-bundle.zip \
  --version 0.1.2 \
  --deployment-id THE_SAVED_DEPLOYMENT_ID \
  --report /absolute/path/to/recovery-report.json
```

Supply the two Central token values through the environment only. The helper
never retries an upload or publish POST. Existing public versions cannot be
overwritten. A successful report requires both Central's `PUBLISHED` state and
public artifact hash verification. Public signed bundles and safe deployment
reports are retained as workflow artifacts for 30 days.

The publishing credentials are configured for this repository. The dedicated
Central token was supplied by the maintainer and verified with a read-only
request against the existing published deployment. No new token was generated
by the setup script. The four secrets' values are never committed or logged.

[Publisher protocol checks](../.github/workflows/android-publisher-checks.yml)
run separately when the publishing helpers change, using mocked requests and no
credentials. Completed Central deployments may return an empty package list;
this is accepted only for `PUBLISHED`, with exact deployment identity and all
public artifact hashes still required. Pre-publication identity checks remain
strict.
