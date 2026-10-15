"""Replay the pinned fixture corpus and require its exact known mismatch."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

EXPECTED_SUMMARY = {"total": 227, "passed": 226, "failed": 1, "unsupported": 0}
KNOWN_FAILURE = {
    "file": "valid_blocks/highGasUsage.json",
    "name": "./updated_2/highGasUsage.json::highGasUsage_MONAD_EIGHT",
    "network": "MONAD_EIGHT",
    "status": "failed",
    "message": (
        "block 1: receipts root mismatch: expected "
        "0x0ea87bd79fe41df810ea2d14ab13c6a3095c5ac9f181a747d9eccca4df09a392, got "
        "0x71ddce3aeced8691b6b4a222fdf0249f8016197a484b54fc5915793fda7b67eb; "
        "transactions [0: halt (ReserveBalanceViolation)]"
    ),
}


def check_report(report, fixture_root, runner_exit_code):
    """Reject regressions, incomplete reports, and changes to the known outcome."""
    if not isinstance(report, dict) or (
        report.get("schema_version") != 1
        or report.get("verification") != "valid_block_execution"
    ):
        raise ValueError("unsupported fixture report schema or verification scope")

    cases = report.get("cases")
    if not isinstance(cases, list) or len(cases) != EXPECTED_SUMMARY["total"]:
        raise ValueError(
            f"expected exactly {EXPECTED_SUMMARY['total']} fixture results"
        )

    root = fixture_root.resolve()
    identities = set()
    known_seen = False
    for case in cases:
        if not isinstance(case, dict) or not all(
            isinstance(case.get(field), str) for field in KNOWN_FAILURE
        ):
            raise ValueError("invalid fixture result")
        path = Path(case["file"])
        if not path.is_absolute():
            raise ValueError(f"fixture path must be absolute: {path}")
        relative = path.resolve().relative_to(root).as_posix()
        identity = (relative, case["name"])
        if identity in identities:
            raise ValueError(f"duplicate fixture result: {identity}")
        identities.add(identity)

        if identity == (KNOWN_FAILURE["file"], KNOWN_FAILURE["name"]):
            if {**case, "file": relative} != KNOWN_FAILURE:
                raise ValueError(f"known highGasUsage outcome changed: {case}")
            known_seen = True
        elif case["status"] != "passed":
            raise ValueError(f"unexpected fixture outcome: {case}")

    if not known_seen:
        raise ValueError("missing known highGasUsage result")
    if report.get("summary") != EXPECTED_SUMMARY:
        raise ValueError(f"unexpected fixture summary: {report.get('summary')}")
    if runner_exit_code != 1:
        raise ValueError(
            f"runner exited {runner_exit_code}; expected 1 for the known mismatch"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("wb") as output:
            result = subprocess.run(
                [
                    str(args.runner.resolve()),
                    str(args.fixtures.resolve()),
                    "--json",
                    "--fail-on-unsupported",
                ],
                stdout=output,
                check=False,
            )
        report = json.loads(args.report.read_text(encoding="utf-8"))
        check_report(report, args.fixtures, result.returncode)
    except (OSError, ValueError) as error:
        message = f"Fixture gate failed: {error}"
        exit_code = 1
    else:
        message = (
            f"Fixture gate passed: {report['summary']['passed']} passed, "
            "1 known highGasUsage mismatch, 0 unsupported."
        )
        exit_code = 0

    print(message, file=sys.stderr if exit_code else sys.stdout)
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(summary).open("a", encoding="utf-8") as output:
            output.write(f"### Execution fixtures\n\n{message}\n")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
