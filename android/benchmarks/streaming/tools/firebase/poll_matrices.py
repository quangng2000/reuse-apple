#!/usr/bin/env python3
"""Read the exact existing Test Lab matrices in an execution plan; never submit tests."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    project = plan["project"]
    if not re.fullmatch(r"[a-z][a-z0-9-]{4,61}[a-z0-9]", project):
        raise ValueError("Invalid project ID")
    requests = plan["requests"]
    if not 1 <= len(requests) <= plan["maxTestExecutions"] <= 2:
        raise ValueError("Unexpected execution count")
    for item in requests:
        if item["status"] != "submitted" or not re.fullmatch(r"matrix-[a-z0-9]+", item["matrixId"]):
            raise ValueError("Only already-submitted, identified matrices may be read")
        if item["kind"] not in {"baseline", "candidate", "profile"}:
            raise ValueError("Unexpected execution label")
    # Capture the short-lived token in memory only; never include it in arguments or logs.
    token = subprocess.check_output(["gcloud", "auth", "print-access-token"], text=True).strip()

    def read(item):
        url = f"https://testing.googleapis.com/v1/projects/{project}/testMatrices/{item['matrixId']}"
        request = Request(url, headers={"Authorization": "Bearer " + token})
        with urlopen(request, timeout=30) as response:
            data = json.load(response)
        if data["testMatrixId"] != item["matrixId"] or data["projectId"] != project:
            raise ValueError("Matrix identity differs from the plan")
        for execution in data.get("testExecutions", []):
            device = execution["environment"]["androidDevice"]
            if device["androidModelId"] != plan["device"]["model"] or device["androidVersionId"] != plan["device"]["version"]:
                raise ValueError("Matrix device differs from the plan")
        destination = args.plan.parent / f"firebase-{item['kind']}-matrix.json"
        destination.write_text(json.dumps(data, indent=2) + "\n")
        return {
            "kind": item["kind"], "matrixId": data["testMatrixId"], "state": data["state"],
            "resultStorage": data.get("resultStorage"),
            "invalidMatrixDetails": data.get("invalidMatrixDetails"),
            "executions": [
                {"id": x.get("id"), "state": x.get("state"),
                 "progress": x.get("testDetails", {}).get("progressMessages", [])[-3:],
                 "error": x.get("testDetails", {}).get("errorMessage"),
                 "issues": x.get("testDetails", {}).get("testIssues", [])}
                for x in data.get("testExecutions", [])
            ],
        }

    with ThreadPoolExecutor(max_workers=2) as pool:
        matrices = list(pool.map(read, requests))
    summary = {"observedUtc": datetime.now(timezone.utc).isoformat(), "matrices": matrices}
    (args.plan.parent / "firebase-observation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
