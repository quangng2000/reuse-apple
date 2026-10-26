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


def make_bundle(path, version=VERSION, corrupt_checksum=False, extra_file=False):
    base = publisher.ARTIFACT + "-" + version
    prefix = publisher.GROUP.replace(".", "/") + "/" + publisher.ARTIFACT + "/" + version + "/"
    pom = (f'<project xmlns="http://maven.apache.org/POM/4.0.0"><modelVersion>4.0.0</modelVersion>'
           f'<groupId>{publisher.GROUP}</groupId><artifactId>{publisher.ARTIFACT}</artifactId>'
           f'<version>{version}</version><packaging>aar</packaging></project>').encode()
    primary = {base + ".pom": pom, base + ".aar": b"compiled Android artifact",
               base + "-sources.jar": b"public Kotlin and native sources",
               base + "-javadoc.jar": b"public API documentation"}
    with zipfile.ZipFile(path, "w") as archive:
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
    return publisher.load_bundle(path, version)


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
        purl = f"pkg:maven/{publisher.GROUP}/{publisher.ARTIFACT}@{VERSION}"
        result = {"deploymentId": DEPLOYMENT, "deploymentName": publisher.ARTIFACT + "-" + VERSION,
                  "deploymentState": state, "purls": [purl + "?type=aar"] if state in {"VALIDATED", "PUBLISHING", "PUBLISHED"} else []}
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


if __name__ == "__main__":
    unittest.main()
