"""Regression coverage for the fixture CI acceptance policy."""

import io
import json
import os
import subprocess
import unittest
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import fixture_ci


class FixtureGateTests(unittest.TestCase):
    def setUp(self):
        self.root = Path("fixture-corpus").resolve()
        self.report = {
            "schema_version": 1,
            "verification": "valid_block_execution",
            "summary": {"total": 227, "passed": 226, "failed": 1, "unsupported": 0},
            "cases": [
                {
                    "file": str(self.root / f"valid_blocks/case-{index}.json"),
                    "name": f"case-{index}_MONAD_EIGHT",
                    "network": "MONAD_EIGHT",
                    "status": "passed",
                    "message": "checked 1 blocks and 1 transactions",
                }
                for index in range(226)
            ]
            + [
                {
                    **fixture_ci.KNOWN_FAILURE,
                    "file": str(self.root / fixture_ci.KNOWN_FAILURE["file"]),
                }
            ],
        }

    def check(self, report=None, exit_code=1):
        fixture_ci.check_report(
            self.report if report is None else report, self.root, exit_code
        )

    def test_accepts_exact_known_failure_without_changing_report(self):
        original = deepcopy(self.report)
        self.check()
        self.assertEqual(self.report, original)

    def test_rejects_other_failures_and_unsupported_cases(self):
        for status in ["failed", "unsupported", "unknown"]:
            with self.subTest(status=status):
                report = deepcopy(self.report)
                report["cases"][0]["status"] = status
                report["cases"][0]["message"] = fixture_ci.KNOWN_FAILURE["message"]
                with self.assertRaises(ValueError):
                    self.check(report)

    def test_rejects_any_changed_known_outcome(self):
        changes = {
            "status": ["passed", "unsupported"],
            "network": ["MONAD_NINE"],
            "name": ["highGasUsage_MONAD_EIGHT"],
            "file": [str(self.root / "another/highGasUsage.json")],
            "message": [
                fixture_ci.KNOWN_FAILURE["message"].replace("block 1:", "block 0:"),
                fixture_ci.KNOWN_FAILURE["message"].replace("0x0ea87", "0x1ea87"),
                fixture_ci.KNOWN_FAILURE["message"].replace("0x71ddc", "0x81ddc"),
                fixture_ci.KNOWN_FAILURE["message"].replace(
                    "ReserveBalanceViolation", "OutOfGas"
                ),
            ],
        }
        for field, values in changes.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    report = deepcopy(self.report)
                    report["cases"][-1][field] = value
                    with self.assertRaises(ValueError):
                        self.check(report)

    def test_rejects_empty_missing_extra_and_duplicate_results(self):
        for cases in [
            [],
            self.report["cases"][:-1],
            self.report["cases"] + [self.report["cases"][0]],
            [self.report["cases"][1]] + self.report["cases"][1:],
            self.report["cases"][:-1]
            + [{**self.report["cases"][0], "name": "replacement"}],
        ]:
            with self.subTest(count=len(cases)), self.assertRaises(ValueError):
                self.check({**self.report, "cases": cases})

    def test_rejects_invalid_schema_scope_and_counts(self):
        for field, value in [
            ("schema_version", 2),
            ("verification", "other"),
            ("summary", {"total": 227, "passed": 227, "failed": 0, "unsupported": 0}),
        ]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.check({**self.report, field: value})

    def test_rejects_paths_outside_the_fixture_corpus(self):
        for path in ["relative.json", str(self.root.parent / "outside.json")]:
            with self.subTest(path=path):
                report = deepcopy(self.report)
                report["cases"][0]["file"] = path
                with self.assertRaises(ValueError):
                    self.check(report)

    def test_rejects_invalid_results_and_runner_exit_codes(self):
        report = deepcopy(self.report)
        del report["cases"][0]["status"]
        with self.assertRaises(ValueError):
            self.check(report)
        for exit_code in [0, 2, 101, -9]:
            with self.subTest(exit_code=exit_code), self.assertRaises(ValueError):
                self.check(exit_code=exit_code)

    def test_cli_preserves_raw_reports_and_writes_summary_on_success_and_failure(self):
        good = json.dumps(self.report).encode()
        bad_report = deepcopy(self.report)
        bad_report["cases"][0]["status"] = "failed"
        for payload, runner_status, expected_status in [
            (good, 1, 0),
            (good, 101, 1),
            (json.dumps(bad_report).encode(), 1, 1),
            (b"not JSON", 1, 1),
            (b"", -9, 1),
        ]:
            with (
                self.subTest(
                    runner_status=runner_status, expected_status=expected_status
                ),
                TemporaryDirectory() as directory,
            ):
                report_path = Path(directory) / "report.json"
                summary_path = Path(directory) / "summary.md"

                def execute(
                    command,
                    *,
                    stdout,
                    check,
                    payload=payload,
                    runner_status=runner_status,
                ):
                    self.assertFalse(check)
                    self.assertEqual(
                        command[1:],
                        [str(self.root), "--json", "--fail-on-unsupported"],
                    )
                    stdout.write(payload)
                    return subprocess.CompletedProcess(command, runner_status)

                with (
                    patch("fixture_ci.subprocess.run", side_effect=execute),
                    patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": str(summary_path)}),
                    redirect_stdout(io.StringIO()),
                    redirect_stderr(io.StringIO()),
                ):
                    status = fixture_ci.main(
                        [
                            "--runner",
                            "monad-fixtures",
                            "--fixtures",
                            str(self.root),
                            "--report",
                            str(report_path),
                        ]
                    )
                self.assertEqual(status, expected_status)
                self.assertEqual(report_path.read_bytes(), payload)
                outcome = "failed" if expected_status else "passed"
                self.assertIn(f"Fixture gate {outcome}:", summary_path.read_text())


if __name__ == "__main__":
    unittest.main()
