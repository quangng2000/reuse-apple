"""Offline publication-state tests. All HTTP and time behavior is simulated."""

import base64
from contextlib import redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
import zipfile

import central_publish as publisher


VERSION = "0.1.2"
DEPLOYMENT = "64789b82-b6d7-44d9-a518-20bbe778a292"
OTHER_DEPLOYMENT = "64789b82-b6d7-44d9-a518-20bbe778a293"


def make_bundle(path, version=VERSION, corrupt_checksum=False, extra_file=False, include_streaming=False):
    artifact_ids = publisher.ARTIFACT_IDS if include_streaming else (publisher.ARTIFACT,)
    with zipfile.ZipFile(path, "w") as archive:
        for artifact_id in artifact_ids:
            base = artifact_id + "-" + version
            prefix = publisher.GROUP.replace(".", "/") + "/" + artifact_id + "/" + version + "/"
            pom = (f'<project xmlns="http://maven.apache.org/POM/4.0.0"><modelVersion>4.0.0</modelVersion>'
                   f'<groupId>{publisher.GROUP}</groupId><artifactId>{artifact_id}</artifactId>'
                   f'<version>{version}</version><packaging>aar</packaging></project>').encode()
            primary = {base + ".pom": pom, base + ".aar": b"compiled Android artifact",
                       base + "-sources.jar": b"public Kotlin and native sources",
                       base + "-javadoc.jar": b"public API documentation"}
            if include_streaming:
                primary[base + ".module"] = json.dumps({"component": {"group": publisher.GROUP,
                                                                      "module": artifact_id, "version": version}}).encode()
            for name, data in primary.items():
                archive.writestr(prefix + name, data)
                archive.writestr(prefix + name + ".asc", "-----BEGIN PGP SIGNATURE-----\nfixture\n-----END PGP SIGNATURE-----\n")
                for algorithm in publisher.CHECKSUMS:
                    value = hashlib.new(algorithm, data).hexdigest()
                    if corrupt_checksum and algorithm == "sha256":
                        value = "0" * 64
                    archive.writestr(prefix + name + "." + algorithm, value + "\n")
        if extra_file:
            archive.writestr(prefix + "credentials.txt", b"must not be uploaded")
    return publisher.load_bundle(path, version, include_streaming=include_streaming)


def rewrite_bundle(path, change):
    with zipfile.ZipFile(path) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    change(entries)
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)


class FakeClock:
    def __init__(self):
        self.now = 0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeClient:
    def __init__(self, bundle, states=None):
        self.bundle = bundle
        self.states = list(states or ["PENDING", "VALIDATED", "PUBLISHING", "PUBLISHED"])
        self.calls = []
        self.is_public = False
        self.preexisting = False
        self.mismatch = False
        self.status_override = {}
        self.upload_error = None
        self.publish_error = None

    def public_file(self, relative):
        self.calls.append(("public", relative))
        if not self.is_public:
            return b"existing immutable version" if self.preexisting else None
        if self.mismatch:
            return b"modified or wrong public bytes"
        return self.bundle["primary"][relative.rsplit("/", 1)[1]]

    def upload(self, bundle, version):
        self.calls.append(("upload", version))
        if self.upload_error:
            raise self.upload_error
        return DEPLOYMENT

    def status(self, deployment_id):
        self.calls.append(("status", deployment_id))
        state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
        self.is_public = state == "PUBLISHED"
        version = self.bundle["components"][0]["coordinate"].rsplit(":", 1)[1]
        purls = [f"pkg:maven/{publisher.GROUP}/{item['artifactId']}@{version}?type=aar"
                 for item in self.bundle["components"]]
        result = {"deploymentId": DEPLOYMENT, "deploymentName": publisher.ARTIFACT + "-" + version,
                  "deploymentState": state, "purls": purls if state in {"VALIDATED", "PUBLISHING", "PUBLISHED"} else []}
        result.update(self.status_override)
        return result

    def publish(self, deployment_id):
        self.calls.append(("publish", deployment_id))
        if self.publish_error:
            raise self.publish_error


class FakeResponse:
    def __init__(self, data, status=200):
        self.data, self.status = data, status

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def read(self, count):
        return self.data[:count]


class FakeOpener:
    def __init__(self, responses):
        self.responses, self.requests = list(responses), []

    def open(self, request, timeout):
        self.requests.append((request, timeout))
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class CentralPublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bundle_path = self.root / "release.zip"
        self.bundle = make_bundle(self.bundle_path)
        self.report = self.root / "report.json"
        self.clock = FakeClock()

    def run_publication(self, client, deadline=60, deployment_id=None):
        return publisher.publish_bundle(self.bundle, VERSION, self.report, client, deadline,
                                        deployment_id=deployment_id, clock=self.clock,
                                        sleep=self.clock.sleep, interval=1)

    def call_count(self, client, operation):
        return sum(call[0] == operation for call in client.calls)

    def test_success_publishes_once_and_verifies_every_primary(self):
        client = FakeClient(self.bundle)
        report = self.run_publication(client)
        self.assertTrue(report["published"])
        self.assertEqual(report["status"], "PUBLIC_ARTIFACTS_VERIFIED")
        self.assertEqual(len(report["artifacts"]), 4)
        self.assertEqual(self.call_count(client, "upload"), 1)
        self.assertEqual(self.call_count(client, "publish"), 1)
        self.assertEqual(json.loads(self.report.read_text())["deploymentId"], DEPLOYMENT)

    def test_eventually_consistent_validated_state_does_not_publish_twice(self):
        client = FakeClient(self.bundle, ["VALIDATED", "VALIDATED", "PUBLISHING", "PUBLISHED"])
        self.assertTrue(self.run_publication(client)["published"])
        self.assertEqual(self.call_count(client, "publish"), 1)

    def test_existing_version_refuses_upload(self):
        client = FakeClient(self.bundle)
        client.preexisting = True
        with self.assertRaisesRegex(publisher.PublishError, "version_already_published"):
            self.run_publication(client)
        self.assertEqual(self.call_count(client, "upload"), 0)

    def test_public_preflight_network_failure_is_closed(self):
        client = FakeClient(self.bundle)
        client.public_file = lambda _: (_ for _ in ()).throw(publisher.PublishError("public_read_network_error"))
        with self.assertRaisesRegex(publisher.PublishError, "public_read_network_error"):
            self.run_publication(client)
        self.assertEqual(self.call_count(client, "upload"), 0)

    def test_failed_state_never_publishes_or_echoes_remote_errors(self):
        client = FakeClient(self.bundle, ["VALIDATING", "FAILED"])
        client.status_override["errors"] = {"password": "sensitive-server-value"}
        with self.assertRaisesRegex(publisher.PublishError, "central_validation_or_publication_failed"):
            self.run_publication(client)
        self.assertNotIn("sensitive-server-value", self.report.read_text())
        self.assertEqual(self.call_count(client, "publish"), 0)

    def test_timeout_keeps_deployment_id_for_recovery(self):
        client = FakeClient(self.bundle, ["PUBLISHING"])
        with self.assertRaisesRegex(publisher.PublishError, "publication_timeout"):
            self.run_publication(client, deadline=2)
        self.assertEqual(json.loads(self.report.read_text())["deploymentId"], DEPLOYMENT)
        self.assertEqual(self.call_count(client, "upload"), 1)

    def test_uncertain_upload_is_never_retried(self):
        client = FakeClient(self.bundle)
        client.upload_error = publisher.PublishError("upload_network_error")
        with self.assertRaisesRegex(publisher.PublishError, "upload_network_error"):
            self.run_publication(client)
        self.assertTrue(json.loads(self.report.read_text())["uploadRequested"])
        with self.assertRaisesRegex(publisher.PublishError, "prior_attempt_requires_explicit_recovery"):
            self.run_publication(client)
        self.assertEqual(self.call_count(client, "upload"), 1)

    def test_uncertain_publish_is_not_retried_during_explicit_recovery(self):
        client = FakeClient(self.bundle, ["VALIDATED"])
        client.publish_error = publisher.PublishError("publish_network_error")
        with self.assertRaisesRegex(publisher.PublishError, "publish_network_error"):
            self.run_publication(client)
        client.publish_error = None
        with self.assertRaisesRegex(publisher.PublishError, "publish_outcome_requires_portal_review"):
            self.run_publication(client, deployment_id=DEPLOYMENT)
        self.assertEqual(self.call_count(client, "publish"), 1)

    def test_explicit_recovery_does_not_upload_again(self):
        client = FakeClient(self.bundle, ["PUBLISHING"])
        with self.assertRaisesRegex(publisher.PublishError, "publication_timeout"):
            self.run_publication(client, deadline=1)
        client.states = ["PUBLISHING", "PUBLISHED"]
        result = self.run_publication(client, deployment_id=DEPLOYMENT)
        self.assertTrue(result["published"])
        self.assertEqual(self.call_count(client, "upload"), 1)

    def test_recovery_requires_a_matching_known_receipt(self):
        client = FakeClient(self.bundle)
        with self.assertRaisesRegex(publisher.PublishError, "recovery_requires_matching_receipt"):
            self.run_publication(client, deployment_id=DEPLOYMENT)
        self.assertFalse(client.calls)
        client.upload_error = publisher.PublishError("upload_network_error")
        with self.assertRaises(publisher.PublishError):
            self.run_publication(client)
        with self.assertRaisesRegex(publisher.PublishError, "unknown_upload_outcome_requires_portal_review"):
            self.run_publication(client, deployment_id=DEPLOYMENT)
        self.assertEqual(self.call_count(client, "upload"), 1)

    def test_wrong_deployment_id_is_rejected_before_publish(self):
        client = FakeClient(self.bundle, ["VALIDATED"])
        client.status_override["deploymentId"] = OTHER_DEPLOYMENT
        with self.assertRaisesRegex(publisher.PublishError, "deployment_identity_mismatch"):
            self.run_publication(client)
        self.assertEqual(self.call_count(client, "publish"), 0)

    def test_wrong_name_and_purl_are_rejected_before_publish(self):
        for field, value, expected in [("deploymentName", "other-release", "deployment_name_mismatch"),
                                        ("purls", ["pkg:maven/other/project@1.0.0"], "deployment_coordinate_mismatch")]:
            with self.subTest(field=field):
                self.report.unlink(missing_ok=True)
                client = FakeClient(self.bundle, ["VALIDATED"])
                client.status_override[field] = value
                with self.assertRaisesRegex(publisher.PublishError, expected):
                    self.run_publication(client)
                self.assertEqual(self.call_count(client, "publish"), 0)

    def test_public_hash_mismatch_cannot_be_reported_as_published(self):
        client = FakeClient(self.bundle, ["PUBLISHED"])
        client.mismatch = True
        with self.assertRaisesRegex(publisher.PublishError, "public_artifact_hash_mismatch"):
            self.run_publication(client)
        self.assertFalse(json.loads(self.report.read_text())["published"])

    def test_published_empty_purls_requires_all_public_hashes_before_success(self):
        client = FakeClient(self.bundle, ["PUBLISHED"])
        client.status_override["purls"] = []
        result = self.run_publication(client)
        self.assertTrue(result["published"])
        self.assertEqual({item["name"] for item in result["artifacts"]}, set(self.bundle["primary"]))
        self.assertEqual(self.call_count(client, "public"), len(self.bundle["primary"]) + 1)
        self.assertEqual(self.call_count(client, "publish"), 0)

    def test_published_empty_purls_cannot_hide_public_hash_mismatch(self):
        client = FakeClient(self.bundle, ["PUBLISHED"])
        client.status_override["purls"] = []
        client.mismatch = True
        with self.assertRaisesRegex(publisher.PublishError, "public_artifact_hash_mismatch"):
            self.run_publication(client)
        result = json.loads(self.report.read_text())
        self.assertFalse(result["published"])
        self.assertEqual(self.call_count(client, "publish"), 0)

    def test_published_wrong_present_purl_is_still_rejected(self):
        client = FakeClient(self.bundle, ["PUBLISHED"])
        client.status_override["purls"] = ["pkg:maven/other/project@1.0.0"]
        with self.assertRaisesRegex(publisher.PublishError, "deployment_coordinate_mismatch"):
            self.run_publication(client)
        self.assertFalse(json.loads(self.report.read_text())["published"])
        self.assertEqual(self.call_count(client, "publish"), 0)

    def test_validated_and_publishing_still_require_nonempty_purls(self):
        for state in ("VALIDATED", "PUBLISHING"):
            with self.subTest(state=state):
                self.report.unlink(missing_ok=True)
                client = FakeClient(self.bundle, [state])
                client.status_override["purls"] = []
                with self.assertRaisesRegex(publisher.PublishError, "missing_deployment_coordinates"):
                    self.run_publication(client)
                self.assertFalse(json.loads(self.report.read_text())["published"])
                self.assertEqual(self.call_count(client, "publish"), 0)

    def test_published_empty_purls_still_requires_exact_deployment_identity(self):
        client = FakeClient(self.bundle, ["PUBLISHED"])
        client.status_override.update(purls=[], deploymentId=OTHER_DEPLOYMENT)
        with self.assertRaisesRegex(publisher.PublishError, "deployment_identity_mismatch"):
            self.run_publication(client)
        self.assertFalse(json.loads(self.report.read_text())["published"])
        self.assertEqual(self.call_count(client, "publish"), 0)

    def test_bundle_checksums_and_paths_are_verified_before_upload(self):
        for option, expected in [("corrupt_checksum", "bundle_checksum_mismatch"),
                                  ("extra_file", "incomplete_or_unexpected_bundle_files")]:
            with self.subTest(option=option):
                with self.assertRaisesRegex(publisher.PublishError, expected):
                    make_bundle(self.bundle_path, **{option: True})

    def test_bundle_version_mismatch_is_rejected(self):
        with self.assertRaisesRegex(publisher.PublishError, "invalid_bundle_path"):
            publisher.load_bundle(self.bundle_path, "0.1.3")

    def test_http_methods_statuses_and_auth_scope_match_portal_contract(self):
        opener = FakeOpener([FakeResponse(b"public file"), FakeResponse(DEPLOYMENT.encode(), 201),
                             FakeResponse(b"", 204)])
        client = publisher.CentralClient("user", "secret", 100, clock=self.clock, opener=opener)
        relative = self.bundle["prefix"] + publisher.ARTIFACT + "-" + VERSION + ".pom"
        client.public_file(relative)
        self.assertEqual(client.upload(self.bundle, VERSION), DEPLOYMENT)
        client.publish(DEPLOYMENT)
        self.assertIsNone(opener.requests[0][0].get_header("Authorization"))
        authorization = "Bearer " + base64.b64encode(b"user:secret").decode()
        self.assertEqual(opener.requests[1][0].get_header("Authorization"), authorization)
        self.assertEqual(opener.requests[1][0].get_method(), "POST")
        self.assertIn("publishingType=USER_MANAGED", opener.requests[1][0].full_url)
        self.assertEqual(opener.requests[2][0].get_method(), "POST")
        self.assertTrue(all(timeout <= 60 for _, timeout in opener.requests))

    def test_status_uses_authenticated_post(self):
        payload = {"deploymentId": DEPLOYMENT, "deploymentName": "fixture", "deploymentState": "PENDING"}
        opener = FakeOpener([FakeResponse(json.dumps(payload).encode())])
        client = publisher.CentralClient("user", "secret", 100, clock=self.clock, opener=opener)
        self.assertEqual(client.status(DEPLOYMENT), payload)
        request = opener.requests[0][0]
        self.assertEqual(request.get_method(), "POST")
        self.assertIn("/status?id=" + DEPLOYMENT, request.full_url)

    def test_redirects_and_untrusted_hosts_cannot_receive_authentication(self):
        opener = FakeOpener([])
        client = publisher.CentralClient("user", "secret", 100, clock=self.clock, opener=opener)
        for url in ["http://central.sonatype.com/api/v1/publisher/upload", "https://attacker.invalid/upload",
                    "https://central.sonatype.com.attacker.invalid/upload"]:
            with self.assertRaisesRegex(publisher.PublishError, "request_destination_refused"):
                client._request(url, authenticated=True)
        self.assertFalse(opener.requests)
        with self.assertRaisesRegex(publisher.PublishError, "http_redirect_refused"):
            publisher.NoRedirect().redirect_request(None, None, 302, "secret", {}, "https://attacker.invalid/?secret")

    def test_network_and_http_errors_do_not_echo_response_or_credentials(self):
        errors = [urllib.error.URLError("password=secret"),
                  urllib.error.HTTPError(publisher.API_ORIGIN, 403, "password=secret", {}, io.BytesIO(b"password=secret"))]
        for error in errors:
            client = publisher.CentralClient("user", "secret", 100, clock=self.clock,
                                             opener=FakeOpener([error]))
            with self.assertRaises(publisher.PublishError) as raised:
                client.upload(self.bundle, VERSION)
            self.assertNotIn("secret", str(raised.exception))

    def test_cli_missing_credentials_does_not_print_environment_secrets(self):
        output = io.StringIO()
        with patch.dict("os.environ", {"CENTRAL_TOKEN_USERNAME": "", "CENTRAL_TOKEN_PASSWORD": "private-value"}, clear=False), redirect_stderr(output):
            code = publisher.main(["--bundle", str(self.bundle_path), "--version", VERSION,
                                   "--report", str(self.report)])
        self.assertEqual(code, 1)
        self.assertNotIn("private-value", output.getvalue())
        self.assertIn("missing_central_credentials", output.getvalue())


class CohortPublicationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bundle_path = self.root / "cohort.zip"
        self.bundle = make_bundle(self.bundle_path, include_streaming=True)
        self.report = self.root / "report.json"
        self.clock = FakeClock()

    run_publication = CentralPublicationTests.run_publication
    call_count = CentralPublicationTests.call_count

    def test_cohort_has_ten_primaries_and_one_upload_publish(self):
        client = FakeClient(self.bundle)
        result = self.run_publication(client)
        self.assertTrue(result["published"])
        self.assertEqual(result["coordinates"], [f"{publisher.GROUP}:{artifact}:{VERSION}" for artifact in publisher.ARTIFACT_IDS])
        self.assertEqual(len(result["artifacts"]), 10)
        self.assertEqual(self.call_count(client, "upload"), 1)
        self.assertEqual(self.call_count(client, "publish"), 1)
        self.assertEqual(self.call_count(client, "public"), 12)
        with zipfile.ZipFile(self.bundle_path) as archive:
            self.assertEqual(len(archive.namelist()), 60)

    def test_cohort_requires_explicit_opt_in_and_both_complete_components(self):
        with self.assertRaises(publisher.PublishError):
            publisher.load_bundle(self.bundle_path, VERSION)
        for component in publisher.ARTIFACT_IDS:
            with self.subTest(component=component):
                make_bundle(self.bundle_path, include_streaming=True)
                rewrite_bundle(self.bundle_path, lambda entries: [entries.pop(name) for name in list(entries)
                                                                  if "/" + component + "/" in name])
                with self.assertRaisesRegex(publisher.PublishError, "incomplete_or_unexpected_bundle_files"):
                    publisher.load_bundle(self.bundle_path, VERSION, include_streaming=True)

    def test_cohort_requires_module_signature_and_checksum_for_each_component(self):
        prefix = self.bundle["components"][1]["prefix"] + publisher.STREAMING_ARTIFACT + "-" + VERSION
        for suffix in (".module", ".aar.asc", ".pom.sha512"):
            with self.subTest(suffix=suffix):
                make_bundle(self.bundle_path, include_streaming=True)
                rewrite_bundle(self.bundle_path, lambda entries: entries.pop(prefix + suffix))
                with self.assertRaisesRegex(publisher.PublishError, "incomplete_or_unexpected_bundle_files"):
                    publisher.load_bundle(self.bundle_path, VERSION, include_streaming=True)

    def test_cohort_rejects_other_versions_unknown_artifacts_and_pom_coordinates(self):
        original = self.bundle_path.read_bytes()
        for replacement in ("/reuse-streaming-android/0.1.3/", "/other-android/0.1.2/"):
            self.bundle_path.write_bytes(original)
            def move(entries):
                for name in list(entries):
                    if "/reuse-streaming-android/0.1.2/" in name:
                        entries[name.replace("/reuse-streaming-android/0.1.2/", replacement)] = entries.pop(name)
            rewrite_bundle(self.bundle_path, move)
            with self.assertRaisesRegex(publisher.PublishError, "invalid_bundle_path"):
                publisher.load_bundle(self.bundle_path, VERSION, include_streaming=True)
        self.bundle_path.write_bytes(original)
        def wrong_pom(entries):
            name = self.bundle["components"][1]["prefix"] + publisher.STREAMING_ARTIFACT + "-" + VERSION + ".pom"
            entries[name] = entries[name].replace(b"<version>0.1.2</version>", b"<version>0.1.3</version>")
            for algorithm in publisher.CHECKSUMS:
                entries[name + "." + algorithm] = hashlib.new(algorithm, entries[name]).hexdigest().encode()
        rewrite_bundle(self.bundle_path, wrong_pom)
        with self.assertRaisesRegex(publisher.PublishError, "bundle_coordinate_mismatch"):
            publisher.load_bundle(self.bundle_path, VERSION, include_streaming=True)

    def test_existing_streaming_version_aborts_before_any_upload(self):
        client = FakeClient(self.bundle)
        public = client.public_file
        client.public_file = lambda relative: b"existing" if "/reuse-streaming-android/" in relative else public(relative)
        with self.assertRaisesRegex(publisher.PublishError, "version_already_published"):
            self.run_publication(client)
        self.assertEqual(self.call_count(client, "upload"), 0)
        self.assertEqual(self.call_count(client, "publish"), 0)

    def test_cohort_requires_exact_complete_purl_set_before_publish(self):
        offline = f"pkg:maven/{publisher.GROUP}/{publisher.ARTIFACT}@{VERSION}"
        streaming = f"pkg:maven/{publisher.GROUP}/{publisher.STREAMING_ARTIFACT}@{VERSION}"
        for purls in ([offline], [streaming], [offline, streaming + "?type=jar"],
                      [offline, streaming, "pkg:maven/other/project@0.1.2"]):
            with self.subTest(purls=purls):
                self.report.unlink(missing_ok=True)
                client = FakeClient(self.bundle, ["VALIDATED"])
                client.status_override["purls"] = purls
                with self.assertRaisesRegex(publisher.PublishError, "deployment_coordinate_mismatch"):
                    self.run_publication(client)
                self.assertEqual(self.call_count(client, "publish"), 0)

    def test_mixed_aar_and_plain_purls_normalize_to_the_exact_cohort(self):
        client = FakeClient(self.bundle)
        client.status_override["purls"] = [f"pkg:maven/{publisher.GROUP}/{publisher.ARTIFACT}@{VERSION}",
                                           f"pkg:maven/{publisher.GROUP}/{publisher.STREAMING_ARTIFACT}@{VERSION}?type=aar"]
        self.assertTrue(self.run_publication(client)["published"])

    def test_published_empty_purls_verifies_all_ten_hashes_without_publish_mutation(self):
        client = FakeClient(self.bundle, ["PUBLISHED"])
        client.status_override["purls"] = []
        result = self.run_publication(client)
        self.assertTrue(result["published"])
        self.assertEqual(len(result["artifacts"]), 10)
        self.assertEqual(self.call_count(client, "publish"), 0)
        self.assertEqual(self.call_count(client, "public"), 12)

    def test_published_partial_purls_and_streaming_hash_mismatch_never_succeed(self):
        client = FakeClient(self.bundle, ["PUBLISHED"])
        client.status_override["purls"] = [f"pkg:maven/{publisher.GROUP}/{publisher.ARTIFACT}@{VERSION}"]
        with self.assertRaisesRegex(publisher.PublishError, "deployment_coordinate_mismatch"):
            self.run_publication(client)
        self.report.unlink()
        client = FakeClient(self.bundle, ["PUBLISHED"])
        client.status_override["purls"] = []
        original_read = client.public_file
        client.public_file = lambda relative: b"wrong streaming bytes" if client.is_public and "/reuse-streaming-android/" in relative else original_read(relative)
        with self.assertRaisesRegex(publisher.PublishError, "public_artifact_hash_mismatch"):
            self.run_publication(client)
        self.assertFalse(json.loads(self.report.read_text())["published"])

    def test_known_cohort_recovery_never_uploads_again(self):
        client = FakeClient(self.bundle, ["PUBLISHING"])
        with self.assertRaisesRegex(publisher.PublishError, "publication_timeout"):
            self.run_publication(client, deadline=1)
        client.states = ["PUBLISHED"]
        self.assertTrue(self.run_publication(client, deployment_id=DEPLOYMENT)["published"])
        self.assertEqual(self.call_count(client, "upload"), 1)
        self.assertEqual(self.call_count(client, "publish"), 0)

    def test_cohort_receipt_cannot_be_recovered_with_incomplete_coordinates(self):
        client = FakeClient(self.bundle, ["PUBLISHING"])
        with self.assertRaises(publisher.PublishError):
            self.run_publication(client, deadline=1)
        prior = json.loads(self.report.read_text())
        prior["coordinates"] = prior["coordinates"][:1]
        self.report.write_text(json.dumps(prior))
        before = list(client.calls)
        with self.assertRaisesRegex(publisher.PublishError, "prior_report_cohort_mismatch"):
            self.run_publication(client, deployment_id=DEPLOYMENT)
        self.assertEqual(before, client.calls)

    def test_uncertain_cohort_mutations_are_never_automatically_retried(self):
        client = FakeClient(self.bundle)
        client.upload_error = publisher.PublishError("upload_network_error")
        with self.assertRaises(publisher.PublishError):
            self.run_publication(client)
        with self.assertRaisesRegex(publisher.PublishError, "unknown_upload_outcome_requires_portal_review"):
            self.run_publication(client, deployment_id=DEPLOYMENT)
        self.assertEqual(self.call_count(client, "upload"), 1)
        self.report.unlink()
        client = FakeClient(self.bundle, ["VALIDATED"])
        client.publish_error = publisher.PublishError("publish_network_error")
        with self.assertRaises(publisher.PublishError):
            self.run_publication(client)
        client.publish_error = None
        with self.assertRaisesRegex(publisher.PublishError, "publish_outcome_requires_portal_review"):
            self.run_publication(client, deployment_id=DEPLOYMENT)
        self.assertEqual(self.call_count(client, "publish"), 1)

    def test_legacy_011_singleton_receipt_remains_recoverable(self):
        version = "0.1.1"
        bundle = make_bundle(self.root / "legacy.zip", version=version)
        client = FakeClient(bundle, ["PUBLISHED"])
        self.report.write_text(json.dumps({"coordinate": f"{publisher.GROUP}:{publisher.ARTIFACT}:{version}",
                                          "bundleSha256": bundle["sha256"], "deploymentId": DEPLOYMENT,
                                          "uploadRequested": True, "publishRequested": True,
                                          "publishAccepted": True, "states": ["PUBLISHED"]}))
        result = publisher.publish_bundle(bundle, version, self.report, client, 60, deployment_id=DEPLOYMENT,
                                          clock=self.clock, sleep=self.clock.sleep)
        self.assertTrue(result["published"])
        self.assertEqual(self.call_count(client, "upload"), 0)
        self.assertEqual(self.call_count(client, "publish"), 0)

    def test_public_reads_reject_artifact_name_aliases_and_unknown_projects(self):
        client = publisher.CentralClient("user", "secret", 60, clock=self.clock, opener=FakeOpener([]))
        for relative in (f"io/github/quangng2000/unknown/{VERSION}/unknown-{VERSION}.pom",
                         f"io/github/quangng2000/{publisher.ARTIFACT}/{VERSION}/{publisher.STREAMING_ARTIFACT}-{VERSION}.pom",
                         f"io/github/quangng2000/{publisher.STREAMING_ARTIFACT}/{VERSION}/../../private.pom"):
            with self.subTest(relative=relative), self.assertRaises(publisher.PublishError):
                client.public_file(relative)


if __name__ == "__main__":
    unittest.main()
