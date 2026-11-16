import argparse
import contextlib
import io
import json
import sys
import tempfile
import types
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

    def assert_cli_error_is_safe(self, result, stderr, expected):
        self.assertEqual(result, 2)
        self.assertIn(expected, stderr)
        self.assertNotIn("Traceback", stderr)
        self.assertNotIn("private detail", stderr)
        self.assertNotIn("C:\\secret", stderr)

    def run_with_interpreter(self, directory, interpreter, bench="0"):
        model = Path(directory) / "model.tflite"
        model.write_bytes(b"test model")
        stderr = io.StringIO()
        with patch.object(sys, "argv", ["inspect_model.py", str(model),
                                         "--bench", bench]), \
                patch.object(inspect_model, "load_interpreter",
                             return_value=(interpreter, "test-backend")), \
                contextlib.redirect_stderr(stderr):
            result = inspect_model.main()
        return result, stderr.getvalue()

    def test_missing_interpreter_has_install_guidance(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.tflite"
            model.write_bytes(b"test model")
            stderr = io.StringIO()
            with patch.object(sys, "argv", ["inspect_model.py", str(model)]), \
                    patch.object(inspect_model, "load_interpreter",
                                 side_effect=inspect_model.InterpreterUnavailable(
                                     "No TFLite interpreter is installed.")), \
                    contextlib.redirect_stderr(stderr):
                result = inspect_model.main()
        self.assertEqual(result, 2)
        self.assertIn("No TFLite interpreter is installed", stderr.getvalue())

    def test_interpreter_construction_error_is_generic_and_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "corrupt.tflite"
            model.write_bytes(b"not a model")
            stderr = io.StringIO()
            with patch.object(sys, "argv", ["inspect_model.py", str(model)]), \
                    patch.object(inspect_model, "load_interpreter",
                                 side_effect=RuntimeError("private detail")), \
                    contextlib.redirect_stderr(stderr):
                result = inspect_model.main()
        self.assert_cli_error_is_safe(result, stderr.getvalue(), "model load failed")

    def test_allocation_error_is_generic_and_safe(self):
        class BrokenInterpreter(FakeInterpreter):
            def allocate_tensors(self):
                raise ValueError("private detail")
        with tempfile.TemporaryDirectory() as directory:
            result, stderr = self.run_with_interpreter(directory, BrokenInterpreter())
        self.assert_cli_error_is_safe(result, stderr, "model inspection failed")

    def test_input_detail_error_is_generic_and_safe(self):
        class BrokenInterpreter(FakeInterpreter):
            def get_input_details(self):
                raise ValueError("private detail")
        with tempfile.TemporaryDirectory() as directory:
            result, stderr = self.run_with_interpreter(directory, BrokenInterpreter())
        self.assert_cli_error_is_safe(result, stderr, "model inspection failed")

    def test_invalid_quantization_metadata_is_rejected(self):
        class BrokenMetadataInterpreter(FakeInterpreter):
            def get_input_details(self):
                detail = self.tensor_detail("images")
                detail["quantization"] = (0.0,)
                return [detail]
        with tempfile.TemporaryDirectory() as directory:
            result, stderr = self.run_with_interpreter(directory, BrokenMetadataInterpreter())
        self.assert_cli_error_is_safe(result, stderr, "model inspection failed")

    def test_empty_tensor_lists_are_rejected(self):
        class EmptyInputs(FakeInterpreter):
            def get_input_details(self):
                return []
        with tempfile.TemporaryDirectory() as directory:
            result, stderr = self.run_with_interpreter(directory, EmptyInputs())
        self.assert_cli_error_is_safe(result, stderr, "model inspection failed")

    def test_benchmark_invocation_error_is_generic_and_safe(self):
        class BrokenInterpreter(FakeInterpreter):
            def get_input_details(self):
                detail = self.tensor_detail("images")
                detail["index"] = 0
                return [detail]
            def invoke(self):
                raise RuntimeError("private detail")
        class FakeInput:
            def astype(self, dtype):
                return self
        fake_numpy = types.SimpleNamespace(
            random=types.SimpleNamespace(rand=lambda *shape: FakeInput()))
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.tflite"
            model.write_bytes(b"test model")
            stderr = io.StringIO()
            with patch.object(sys, "argv", ["inspect_model.py", str(model),
                                             "--bench", "1"]), \
                    patch.object(inspect_model, "load_interpreter",
                                 return_value=(BrokenInterpreter(), "test-backend")), \
                    patch.dict(sys.modules, {"numpy": fake_numpy}), \
                    contextlib.redirect_stderr(stderr):
                result = inspect_model.main()
        self.assert_cli_error_is_safe(result, stderr.getvalue(), "benchmark failed")

    def test_valid_benchmark_preserves_latency_report_schema(self):
        class BenchmarkInterpreter(FakeInterpreter):
            def get_input_details(self):
                detail = self.tensor_detail("images")
                detail["index"] = 0
                return [detail]

            def set_tensor(self, index, value):
                pass

            def invoke(self):
                pass

        class FakeInput:
            def astype(self, dtype):
                return self

        fake_numpy = types.SimpleNamespace(
            random=types.SimpleNamespace(rand=lambda *shape: FakeInput()))
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.tflite"
            model.write_bytes(b"test model")
            stdout = io.StringIO()
            with patch.object(sys, "argv", ["inspect_model.py", str(model),
                                             "--bench", "1"]), \
                    patch.object(inspect_model, "load_interpreter",
                                 return_value=(BenchmarkInterpreter(), "test-backend")), \
                    patch.dict(sys.modules, {"numpy": fake_numpy}), \
                    contextlib.redirect_stdout(stdout):
                result = inspect_model.main()
        report = json.loads(stdout.getvalue())
        self.assertEqual(result, 0)
        self.assertEqual(set(report["host_latency_ms"]), {
            "median", "min", "max", "runs", "note",
        })
        self.assertEqual(report["host_latency_ms"]["runs"], 1)

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
        self.assertEqual(set(report), {
            "file", "bytes", "sha256", "interpreter", "inputs", "outputs",
            "tensor_count",
        })


if __name__ == "__main__":
    unittest.main()
