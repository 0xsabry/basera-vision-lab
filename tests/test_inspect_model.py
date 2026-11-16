import argparse
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import inspect_model


class FakeInterpreter:
    @staticmethod
    def tensor_detail(name):
        return {
            "name": name,
            "shape": [1, 3, 640, 640],
            "dtype": "float32",
            "quantization": (0.0, 0),
        }

    def allocate_tensors(self):
        pass

    def get_input_details(self):
        return [self.tensor_detail("images")]

    def get_output_details(self):
        return [self.tensor_detail("output0")]

    def get_tensor_details(self):
        return [self.tensor_detail("images"), self.tensor_detail("output0")]


class InspectModelTests(unittest.TestCase):
    def test_bench_count_accepts_zero_and_positive_values(self):
        self.assertEqual(inspect_model.non_negative_int("0"), 0)
        self.assertEqual(inspect_model.non_negative_int("3"), 3)

    def test_bench_count_rejects_negative_and_non_integer_values(self):
        for value in ("-1", "1.5"):
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    inspect_model.non_negative_int(value)

    def test_directory_path_returns_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            stderr = io.StringIO()
            with patch.object(sys, "argv", ["inspect_model.py", directory]), \
                    contextlib.redirect_stderr(stderr):
                result = inspect_model.main()

        self.assertEqual(result, 1)
        self.assertIn("not a regular file", stderr.getvalue())

    def test_empty_model_file_returns_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "empty.tflite"
            model.touch()
            stderr = io.StringIO()
            with patch.object(sys, "argv", ["inspect_model.py", str(model)]), \
                    contextlib.redirect_stderr(stderr):
                result = inspect_model.main()

        self.assertEqual(result, 1)
        self.assertIn("model is empty", stderr.getvalue())

    def test_negative_benchmark_is_rejected_by_argument_parser(self):
        stderr = io.StringIO()
        with patch.object(sys, "argv", ["inspect_model.py", "model.tflite",
                                         "--bench", "-1"]), \
                contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                inspect_model.main()

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("must be zero or greater", stderr.getvalue())

    def test_missing_interpreter_has_install_guidance(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.tflite"
            model.write_bytes(b"test model")
            stderr = io.StringIO()
            with patch.object(sys, "argv", ["inspect_model.py", str(model)]), \
                    patch.object(inspect_model, "load_interpreter",
                                 side_effect=RuntimeError("No TFLite interpreter is installed.")), \
                    contextlib.redirect_stderr(stderr):
                result = inspect_model.main()

        self.assertEqual(result, 2)
        self.assertIn("No TFLite interpreter is installed", stderr.getvalue())

    def test_model_loader_error_is_reported_without_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "corrupt.tflite"
            model.write_bytes(b"not a model")
            stderr = io.StringIO()
            with patch.object(sys, "argv", ["inspect_model.py", str(model)]), \
                    patch.object(inspect_model, "load_interpreter",
                                 side_effect=ValueError("invalid FlatBuffer")), \
                    contextlib.redirect_stderr(stderr):
                result = inspect_model.main()

        self.assertEqual(result, 2)
        self.assertIn("model load failed: invalid FlatBuffer", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_interpreter_inspection_error_is_reported_without_traceback(self):
        class BrokenInterpreter(FakeInterpreter):
            def allocate_tensors(self):
                raise ValueError("invalid tensor table")

        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "corrupt.tflite"
            model.write_bytes(b"not a model")
            stderr = io.StringIO()
            with patch.object(sys, "argv", ["inspect_model.py", str(model)]), \
                    patch.object(inspect_model, "load_interpreter",
                                 return_value=(BrokenInterpreter(), "test-backend")), \
                    contextlib.redirect_stderr(stderr):
                result = inspect_model.main()

        self.assertEqual(result, 2)
        self.assertIn("model inspection failed: invalid tensor table", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_valid_file_emits_json_report(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.tflite"
            model.write_bytes(b"test model")
            stdout = io.StringIO()
            with patch.object(sys, "argv", ["inspect_model.py", str(model)]), \
                    patch.object(inspect_model, "load_interpreter",
                                 return_value=(FakeInterpreter(), "test-backend")), \
                    contextlib.redirect_stdout(stdout):
                result = inspect_model.main()

        report = json.loads(stdout.getvalue())
        self.assertEqual(result, 0)
        self.assertEqual(report["interpreter"], "test-backend")
        self.assertEqual(report["tensor_count"], 2)
        self.assertEqual(report["inputs"][0]["shape"], [1, 3, 640, 640])


if __name__ == "__main__":
    unittest.main()
