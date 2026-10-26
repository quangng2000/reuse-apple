#!/usr/bin/env python3
"""Read-only recovery of one already accepted Central publication.

The original signed ZIP, its independent SHA-256 and the matching durable
accepted-publish receipt are required. This command cannot upload or request
publication: it only reads the known deployment status and verifies every
primary artifact's public repository bytes. Credentials are read exclusively
from CENTRAL_TOKEN_USERNAME/CENTRAL_TOKEN_PASSWORD.
"""

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
import urllib.parse

import central_publish as publisher


MAX_RECEIPT_BYTES = 1024 * 1024


class ReadOnlyCentralClient(publisher.CentralClient):
    """Mutation guards remain active even if the publication helper changes."""

    def __init__(self, username, password, deadline, *, deployment_id, public_paths, **kwargs):
        self._deployment_id = publisher.validate_deployment_id(deployment_id)
        self._public_paths = frozenset("/maven2/" + path for path in public_paths)
        publisher.require(self._public_paths, "verification_public_paths_required")
        super().__init__(username, password, deadline, **kwargs)

    def upload(self, bundle, version):
        raise publisher.PublishError("read_only_upload_refused")

    def publish(self, deployment_id):
        raise publisher.PublishError("read_only_publish_refused")

    def _request(self, url, method="GET", data=None, headers=None, authenticated=False,
                 missing_ok=False, operation="request"):
        parts = urllib.parse.urlsplit(url)
        if operation == "status":
            try:
                query = urllib.parse.parse_qs(parts.query, strict_parsing=True)
            except ValueError:
                raise publisher.PublishError("read_only_request_refused") from None
            publisher.require(method == "POST" and authenticated is True and data == b""
                              and parts.scheme == "https" and parts.netloc == "central.sonatype.com"
                              and parts.path == "/api/v1/publisher/status" and not parts.fragment
                              and query == {"id": [self._deployment_id]},
                              "read_only_request_refused")
        elif operation == "public_read":
            publisher.require(method == "GET" and authenticated is False and data is None
                              and parts.scheme == "https" and parts.netloc == "repo.maven.apache.org"
                              and parts.path in self._public_paths and not parts.query and not parts.fragment,
                              "read_only_request_refused")
        else:
            raise publisher.PublishError("read_only_request_refused")
        return super()._request(url, method, data, headers, authenticated, missing_ok, operation)


def require_accepted_receipt(report_path, bundle_path, bundle, version, deployment_id):
    """Validate all recovery authority before a client is constructed or used."""
    publisher.require(report_path.is_file() and not report_path.is_symlink(),
                      "verification_requires_accepted_receipt")
    publisher.require(report_path.resolve() != bundle_path.resolve()
                      and not report_path.samefile(bundle_path), "report_must_not_overwrite_bundle")
    publisher.require(0 < report_path.stat().st_size <= MAX_RECEIPT_BYTES,
                      "verification_receipt_size_limit")
    try:
        receipt = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        raise publisher.PublishError("verification_receipt_unreadable") from None
    publisher.require(isinstance(receipt, dict), "verification_receipt_invalid")
    # Version one was a singleton offline receipt; cohort releases require v2.
    schemas = (2,) if len(bundle["coordinates"]) > 1 else (1, 2)
    publisher.require(type(receipt.get("schemaVersion")) is int and receipt["schemaVersion"] in schemas
                      and receipt.get("publishingType") == "USER_MANAGED",
                      "verification_receipt_invalid")
    expected_coordinate = publisher.GROUP + ":" + publisher.ARTIFACT + ":" + version
    publisher.require(receipt.get("coordinate") == expected_coordinate
                      and receipt.get("bundleSha256") == bundle["sha256"],
                      "verification_receipt_artifact_mismatch")
    coordinates = receipt.get("coordinates", [receipt.get("coordinate")])
    publisher.require(coordinates == bundle["coordinates"], "verification_receipt_cohort_mismatch")
    publisher.require(receipt.get("deploymentId") == deployment_id, "verification_receipt_deployment_mismatch")
    publisher.require(all(receipt.get(name) is True for name in
                          ("uploadRequested", "publishRequested", "publishAccepted")),
                      "verification_requires_accepted_publish")
    states = receipt.get("states", [])
    publisher.require(isinstance(states, list) and len(states) <= 4096
                      and all(isinstance(state, str) and state in publisher.STATES for state in states),
                      "verification_receipt_state_invalid")
    return receipt


def verify_publication(bundle_path, version, deployment_id, expected_bundle_sha256, report_path,
                       include_streaming=False, timeout_seconds=1800, *, client=None,
                       clock=time.monotonic, sleep=time.sleep, interval=15):
    """Verify the exact known deployment without any upload/publish capability."""
    publisher.require(type(timeout_seconds) is int and 60 <= timeout_seconds <= 3600, "timeout_out_of_bounds")
    publisher.validate_deployment_id(deployment_id)
    publisher.require(isinstance(expected_bundle_sha256, str)
                      and re.fullmatch(r"[0-9a-f]{64}", expected_bundle_sha256), "invalid_expected_bundle_sha256")
    bundle_path, report_path = Path(bundle_path), Path(report_path)
    bundle = publisher.load_bundle(bundle_path, version, include_streaming=include_streaming)
    publisher.require(bundle["sha256"] == expected_bundle_sha256, "verification_bundle_hash_mismatch")
    require_accepted_receipt(report_path, bundle_path, bundle, version, deployment_id)
    deadline = clock() + timeout_seconds
    if client is None:
        client = ReadOnlyCentralClient(os.environ.get("CENTRAL_TOKEN_USERNAME", ""),
                                       os.environ.get("CENTRAL_TOKEN_PASSWORD", ""), deadline, clock=clock,
                                       deployment_id=deployment_id, public_paths=bundle["paths"].values())
    publisher.require(isinstance(client, ReadOnlyCentralClient), "verification_requires_read_only_client")
    report = publisher.publish_bundle(bundle, version, report_path, client, deadline,
                                      deployment_id=deployment_id, clock=clock, sleep=sleep, interval=interval)
    states = report.get("states")
    publisher.require(report.get("published") is True and report.get("status") == "PUBLIC_ARTIFACTS_VERIFIED"
                      and report.get("deploymentId") == deployment_id
                      and report.get("bundleSha256") == expected_bundle_sha256
                      and report.get("coordinates") == bundle["coordinates"]
                      and isinstance(states, list) and states and states[-1] == "PUBLISHED",
                      "verification_incomplete")
    artifacts = report.get("artifacts")
    publisher.require(isinstance(artifacts, list) and len(artifacts) == len(bundle["primary"]),
                      "verification_incomplete_public_hashes")
    names = set()
    for item in artifacts:
        publisher.require(isinstance(item, dict) and isinstance(item.get("name"), str),
                          "verification_incomplete_public_hashes")
        name = item["name"]
        publisher.require(name in bundle["primary"] and name not in names,
                          "verification_incomplete_public_hashes")
        contents = bundle["primary"][name]
        publisher.require(type(item.get("bytes")) is int and item["bytes"] == len(contents)
                          and item.get("sha256") == publisher.sha256(contents),
                          "verification_incomplete_public_hashes")
        names.add(name)
    publisher.require(names == set(bundle["primary"]), "verification_incomplete_public_hashes")
    return {"coordinates": bundle["coordinates"], "deploymentId": deployment_id,
            "bundleSha256": expected_bundle_sha256, "status": report["status"],
            "published": True, "artifactCount": len(names), "readOnly": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--deployment-id", required=True)
    parser.add_argument("--expected-bundle-sha256", required=True)
    parser.add_argument("--report", type=Path, required=True,
                        help="Existing accepted-publish receipt; updated with status/hash verification only")
    parser.add_argument("--include-streaming", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args(argv)
    try:
        result = verify_publication(args.bundle, args.version, args.deployment_id,
                                    args.expected_bundle_sha256, args.report,
                                    include_streaming=args.include_streaming, timeout_seconds=args.timeout_seconds)
        print(json.dumps(result, sort_keys=True))
        return 0
    except publisher.PublishError as error:
        print(json.dumps({"ok": False, "error": str(error), "readOnly": True}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
