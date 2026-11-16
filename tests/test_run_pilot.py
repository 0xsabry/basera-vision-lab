import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from pilot import run_pilot


class StageReportTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.original_runs = run_pilot.RUNS
        run_pilot.RUNS = Path(self.temp_dir.name) / "runs"
        self.addCleanup(setattr, run_pilot, "RUNS", self.original_runs)

    def write_result(self, name, content):
        result = run_pilot.RUNS / name / "results.json"
        result.parent.mkdir(parents=True)
        result.write_text(content)
        return result

    @staticmethod
    def valid_result():
        return json.dumps({
            "model": "yolo11n",
            "epochs": 30,
            "epoch_time_median_s": 12.5,
            "total_train_s": 400.0,
            "map50": 0.8,
            "map50_95": 0.6,
            "recall": 0.7,
            "exports_tflite": {
                "fp32": {"bytes": 1000},
                "fp16": {"bytes": 500},
            },
        })

    def run_report(self):
        with contextlib.redirect_stdout(io.StringIO()):
            run_pilot.stage_report()
        return (run_pilot.RUNS / "REPORT.md").read_text()

    def test_report_keeps_valid_results_and_names_skipped_invalid_file(self):
        self.write_result("train-good", self.valid_result())
        bad = self.write_result("train-bad", "{")

        report = self.run_report()

        self.assertIn("| yolo11n | 30 | 12.5s |", report)
        self.assertIn(f"`{bad}:", report)

    def test_report_is_created_when_no_valid_results_exist(self):
        self.write_result("train-bad", "[]")

        report = self.run_report()

        self.assertIn("No valid pilot results found.", report)
        self.assertIn("Skipped invalid result files:", report)

    def test_report_handles_missing_results_directory(self):
        self.assertFalse(run_pilot.RUNS.exists())

        report = self.run_report()

        self.assertIn("No valid pilot results found.", report)


if __name__ == "__main__":
    unittest.main()
