"""Read-only publication recovery tests: no real network or credentials."""

from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import central_publish as publisher
import central_verify as verifier
from test_central_publish import DEPLOYMENT, OTHER_DEPLOYMENT, VERSION, FakeClock, FakeOpener, FakeResponse, make_bundle


class FakeReadOnlyClient(verifier.ReadOnlyCentralClient):
    def __init__(self, bundle, states=None):
        self.bundle = bundle
        self.states = list(states or ["PUBLISHING", "PUBLISHED"])
        self.calls = []
        self.is_public = False
        self.status_override = {}
        self.mismatch = False

    def public_file(self, relative):
        self.calls.append(("public", relative))
        if not self.is_public:
            return None
        return b"wrong public artifact" if self.mismatch else self.bundle["primary"][relative.rsplit("/", 1)[1]]

    def status(self, deployment_id):
        self.calls.append(("status", deployment_id))
        state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
        self.is_public = state == "PUBLISHED"
        version = self.bundle["coordinates"][0].rsplit(":", 1)[1]
        purls = [f"pkg:maven/{publisher.GROUP}/{item['artifactId']}@{version}?type=aar"
                 for item in self.bundle["components"]]
        response = {"deploymentId": DEPLOYMENT, "deploymentName": publisher.ARTIFACT + "-" + version,
                    "deploymentState": state, "purls": purls if state == "VALIDATED" else []}
        response.update(self.status_override)
        return response


class ReadOnlyVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bundle_path = self.root / "original-sixty-entry.zip"
        self.bundle = make_bundle(self.bundle_path, include_streaming=True)
        self.report = self.root / "accepted-publish-receipt.json"
        self.clock = FakeClock()
        self.original_bundle = self.bundle_path.read_bytes()
        self.write_receipt()

    def write_receipt(self, **changes):
        receipt = {"schemaVersion": 2, "coordinate": self.bundle["coordinates"][0],
                   "coordinates": self.bundle["coordinates"], "bundleSha256": self.bundle["sha256"],
                   "publishingType": "USER_MANAGED", "deploymentId": DEPLOYMENT,
                   "uploadRequested": True, "publishRequested": True, "publishAccepted": True,
                   "published": False, "status": "PUBLISHING", "states": ["VALIDATED", "PUBLISHING"],
                   "error": "missing_deployment_coordinates", "artifacts": []}
        receipt.update(changes)
        self.report.write_text(json.dumps(receipt))

    def verify(self, client=None, **options):
        defaults = {"include_streaming": True, "timeout_seconds": 60,
                    "client": client or FakeReadOnlyClient(self.bundle),
                    "clock": self.clock, "sleep": self.clock.sleep, "interval": 1}
        defaults.update(options)
        return verifier.verify_publication(self.bundle_path, VERSION, DEPLOYMENT,
                                            self.bundle["sha256"], self.report, **defaults)

    def assert_unchanged_rejection(self, client, code, **options):
        receipt = self.report.read_bytes()
        with self.assertRaisesRegex(publisher.PublishError, code):
            self.verify(client, **options)
        self.assertEqual(client.calls, [])
        self.assertEqual(self.bundle_path.read_bytes(), self.original_bundle)
        self.assertEqual(self.report.read_bytes(), receipt)

    def test_original_accepted_receipt_empty_publishing_then_published_verifies_all_ten_hashes(self):
        client = FakeReadOnlyClient(self.bundle)
        with patch.object(publisher.CentralClient, "upload", side_effect=AssertionError("no upload")), \
             patch.object(publisher.CentralClient, "publish", side_effect=AssertionError("no publish")):
            result = self.verify(client)
        self.assertTrue(result["readOnly"])
        self.assertTrue(result["published"])
        self.assertEqual(result["artifactCount"], 10)
        self.assertEqual(result["coordinates"], self.bundle["coordinates"])
        receipt = json.loads(self.report.read_text())
        self.assertEqual(len(receipt["artifacts"]), 10)
        self.assertEqual({item["sha256"] for item in receipt["artifacts"]},
                         {publisher.sha256(contents) for contents in self.bundle["primary"].values()})
        self.assertEqual(self.bundle_path.read_bytes(), self.original_bundle)
        self.assertTrue(all(call[0] in {"public", "status"} for call in client.calls))

    def test_already_published_receipt_still_requires_every_public_hash(self):
        self.write_receipt(published=True, status="PUBLIC_ARTIFACTS_VERIFIED")
        self.assertEqual(self.verify(FakeReadOnlyClient(self.bundle, ["PUBLISHED"]))["artifactCount"], 10)

    def test_receipt_is_required_before_any_client_call(self):
        self.report.unlink()
        client = FakeReadOnlyClient(self.bundle)
        with self.assertRaisesRegex(publisher.PublishError, "verification_requires_accepted_receipt"):
            self.verify(client)
        self.assertEqual(client.calls, [])
        self.assertFalse(self.report.exists())

    def test_missing_or_nonboolean_acceptance_flags_are_rejected_before_network(self):
        for name in ("uploadRequested", "publishRequested", "publishAccepted"):
            for value in (False, None, 0, 1, "true", [True], {}):
                with self.subTest(name=name, value=value):
                    self.write_receipt(**{name: value})
                    self.assert_unchanged_rejection(FakeReadOnlyClient(self.bundle), "verification_requires_accepted_publish")
            self.write_receipt()
            receipt = json.loads(self.report.read_text()); del receipt[name]
            self.report.write_text(json.dumps(receipt))
            self.assert_unchanged_rejection(FakeReadOnlyClient(self.bundle), "verification_requires_accepted_publish")

    def test_wrong_original_hash_id_version_cohort_or_receipt_schema_is_rejected(self):
        for changes, code in [
            ({"bundleSha256": "0" * 64}, "verification_receipt_artifact_mismatch"),
            ({"deploymentId": OTHER_DEPLOYMENT}, "verification_receipt_deployment_mismatch"),
            ({"coordinate": self.bundle["coordinates"][0] + "wrong"}, "verification_receipt_artifact_mismatch"),
            ({"coordinates": self.bundle["coordinates"][:1]}, "verification_receipt_cohort_mismatch"),
            ({"coordinates": list(reversed(self.bundle["coordinates"]))}, "verification_receipt_cohort_mismatch"),
            ({"schemaVersion": 1}, "verification_receipt_invalid"),
            ({"schemaVersion": True}, "verification_receipt_invalid"),
            ({"publishingType": "AUTOMATIC"}, "verification_receipt_invalid"),
            ({"states": ["secret remote text"]}, "verification_receipt_state_invalid"),
        ]:
            with self.subTest(changes=changes):
                self.write_receipt(**changes)
                self.assert_unchanged_rejection(FakeReadOnlyClient(self.bundle), code)

    def test_independent_expected_bundle_hash_is_required(self):
        for expected, code in [("0" * 64, "verification_bundle_hash_mismatch"),
                               ("not-a-hash", "invalid_expected_bundle_sha256")]:
            client = FakeReadOnlyClient(self.bundle)
            original_report = self.report.read_bytes()
            with self.assertRaisesRegex(publisher.PublishError, code):
                verifier.verify_publication(self.bundle_path, VERSION, DEPLOYMENT, expected,
                                            self.report, include_streaming=True, client=client)
            self.assertEqual(client.calls, [])
            self.assertEqual(self.report.read_bytes(), original_report)

    def test_symlink_oversized_malformed_and_nonobject_receipts_are_rejected(self):
        original = self.root / "receipt.json"
        self.report.rename(original); self.report.symlink_to(original)
        self.assert_unchanged_rejection(FakeReadOnlyClient(self.bundle), "verification_requires_accepted_receipt")
        self.report.unlink()
        for content, code in [("x" * (verifier.MAX_RECEIPT_BYTES + 1), "verification_receipt_size_limit"),
                               ("not json", "verification_receipt_unreadable"),
                               ("[]", "verification_receipt_invalid")]:
            self.report.write_text(content)
            self.assert_unchanged_rejection(FakeReadOnlyClient(self.bundle), code)

    def test_status_identity_and_metadata_rules_remain_strict(self):
        for override, code in [
            ({"deploymentId": OTHER_DEPLOYMENT}, "deployment_identity_mismatch"),
            ({"deploymentName": "wrong"}, "deployment_name_mismatch"),
            ({"purls": ["pkg:maven/other/wrong@0.1.2"]}, "deployment_coordinate_mismatch"),
            ({"deploymentState": "VALIDATED", "purls": []}, "missing_deployment_coordinates"),
        ]:
            with self.subTest(override=override):
                self.write_receipt()
                client = FakeReadOnlyClient(self.bundle); client.status_override = override
                with self.assertRaisesRegex(publisher.PublishError, code):
                    self.verify(client)
                self.assertTrue(all(call[0] in {"public", "status"} for call in client.calls))

    def test_validated_after_accepted_publish_never_republishes(self):
        client = FakeReadOnlyClient(self.bundle, ["VALIDATED", "PUBLISHING", "PUBLISHED"])
        self.assertTrue(self.verify(client)["published"])
        self.assertTrue(all(call[0] in {"public", "status"} for call in client.calls))

    def test_failed_state_and_timeout_do_not_enable_mutations(self):
        for states, code in [(["FAILED"], "central_validation_or_publication_failed"),
                             (["PUBLISHING"], "publication_timeout")]:
            self.write_receipt()
            client = FakeReadOnlyClient(self.bundle, states)
            with self.assertRaisesRegex(publisher.PublishError, code):
                self.verify(client)
            self.assertTrue(all(call[0] in {"public", "status"} for call in client.calls))

    def test_public_hash_mismatch_is_terminal(self):
        client = FakeReadOnlyClient(self.bundle, ["PUBLISHED"]); client.mismatch = True
        with self.assertRaisesRegex(publisher.PublishError, "public_artifact_hash_mismatch"):
            self.verify(client)
        self.assertFalse(json.loads(self.report.read_text())["published"])

    def test_even_changed_helper_cannot_upload_or_republish(self):
        for operation, code in [("upload", "read_only_upload_refused"), ("publish", "read_only_publish_refused")]:
            self.write_receipt()
            client = FakeReadOnlyClient(self.bundle)
            def attempted_mutation(*args, **kwargs):
                if operation == "upload":
                    return args[3].upload(args[0], args[1])
                return args[3].publish(kwargs["deployment_id"])
            with patch.object(publisher, "publish_bundle", side_effect=attempted_mutation):
                with self.assertRaisesRegex(publisher.PublishError, code):
                    self.verify(client)
            self.assertEqual(client.calls, [])

    def test_read_only_client_guards_mutation_before_opener(self):
        opener = FakeOpener([])
        client = verifier.ReadOnlyCentralClient("fixture-user", "fixture-pass", 60, clock=self.clock, opener=opener,
                                                deployment_id=DEPLOYMENT, public_paths=self.bundle["paths"].values())
        with self.assertRaisesRegex(publisher.PublishError, "read_only_upload_refused"):
            client.upload(self.bundle, VERSION)
        with self.assertRaisesRegex(publisher.PublishError, "read_only_publish_refused"):
            client.publish(DEPLOYMENT)
        with self.assertRaisesRegex(publisher.PublishError, "read_only_request_refused"):
            client._request(publisher.API_ORIGIN + "/api/v1/publisher/upload", method="POST", operation="upload")
        self.assertEqual(opener.requests, [])

    def test_request_allowlist_binds_exact_known_deployment_and_public_paths(self):
        opener = FakeOpener([FakeResponse(b"{}"), FakeResponse(b"artifact bytes")])
        client = verifier.ReadOnlyCentralClient("fixture-user", "fixture-pass", 60, clock=self.clock, opener=opener,
                                                deployment_id=DEPLOYMENT, public_paths=self.bundle["paths"].values())
        self.assertEqual(client.status(DEPLOYMENT), {})
        path = next(iter(self.bundle["paths"].values()))
        self.assertEqual(client.public_file(path), b"artifact bytes")
        self.assertEqual(len(opener.requests), 2)
        for url, kwargs in [
            (publisher.API_ORIGIN + "/api/v1/publisher/status?id=" + OTHER_DEPLOYMENT,
             {"method": "POST", "data": b"", "authenticated": True, "operation": "status"}),
            (publisher.API_ORIGIN + "/api/v1/publisher/status?bad-query",
             {"method": "POST", "data": b"", "authenticated": True, "operation": "status"}),
            (publisher.API_ORIGIN + "/api/v1/publisher/status?id=" + DEPLOYMENT,
             {"method": "GET", "authenticated": True, "operation": "status"}),
            (publisher.PUBLIC_ORIGIN + "/maven2/other/artifact.pom", {"operation": "public_read"}),
            (publisher.PUBLIC_ORIGIN + "/maven2/" + path, {"method": "POST", "operation": "public_read"}),
        ]:
            with self.subTest(url=url):
                with self.assertRaisesRegex(publisher.PublishError, "read_only_request_refused"):
                    client._request(url, **kwargs)
        self.assertEqual(len(opener.requests), 2)

    def test_no_partial_public_hash_result_can_claim_success(self):
        def partial(*args, **kwargs):
            return {"published": True, "status": "PUBLIC_ARTIFACTS_VERIFIED", "deploymentId": DEPLOYMENT,
                    "bundleSha256": self.bundle["sha256"], "coordinates": self.bundle["coordinates"],
                    "states": ["PUBLISHED"], "artifacts": []}
        with patch.object(publisher, "publish_bundle", side_effect=partial):
            with self.assertRaisesRegex(publisher.PublishError, "verification_incomplete_public_hashes"):
                self.verify()

    def test_complete_public_hash_result_without_published_state_cannot_claim_success(self):
        report = {"published": True, "status": "PUBLIC_ARTIFACTS_VERIFIED", "deploymentId": DEPLOYMENT,
                  "bundleSha256": self.bundle["sha256"], "coordinates": self.bundle["coordinates"],
                  "states": ["PUBLISHING"], "artifacts": [
                      {"name": name, "bytes": len(contents), "sha256": publisher.sha256(contents)}
                      for name, contents in self.bundle["primary"].items()]}
        with patch.object(publisher, "publish_bundle", return_value=report):
            with self.assertRaisesRegex(publisher.PublishError, "verification_incomplete"):
                self.verify()

    def test_cli_prints_only_safe_read_only_summary(self):
        expected = {"coordinates": self.bundle["coordinates"], "deploymentId": DEPLOYMENT,
                    "bundleSha256": self.bundle["sha256"], "status": "PUBLIC_ARTIFACTS_VERIFIED",
                    "published": True, "artifactCount": 10, "readOnly": True}
        output, errors = io.StringIO(), io.StringIO()
        args = ["--bundle", str(self.bundle_path), "--version", VERSION, "--deployment-id", DEPLOYMENT,
                "--expected-bundle-sha256", self.bundle["sha256"], "--report", str(self.report), "--include-streaming"]
        with patch.object(verifier, "verify_publication", return_value=expected) as call, \
             redirect_stdout(output), redirect_stderr(errors):
            self.assertEqual(verifier.main(args), 0)
        self.assertEqual(json.loads(output.getvalue()), expected)
        self.assertEqual(errors.getvalue(), "")
        self.assertTrue(call.call_args.kwargs["include_streaming"])

    def test_offline_singleton_can_be_read_only_verified(self):
        self.bundle = make_bundle(self.bundle_path, include_streaming=False)
        self.write_receipt()
        result = self.verify(FakeReadOnlyClient(self.bundle), include_streaming=False)
        self.assertEqual(result["artifactCount"], 4)


if __name__ == "__main__":
    unittest.main()
