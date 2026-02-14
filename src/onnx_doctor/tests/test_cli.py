"""Tests for the CLI."""

from __future__ import annotations

import subprocess
import sys
import unittest


class CLITest(unittest.TestCase):
    def _run(self, *args: str) -> subprocess.CompletedProcess:
        env = {**__import__("os").environ, "PYTHONUTF8": "1"}
        return subprocess.run(
            [sys.executable, "-m", "onnx_doctor", *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
            env=env,
        )

    def test_help(self):
        result = self._run("--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("onnx-doctor", result.stdout)

    def test_list_rules(self):
        result = self._run("list-rules")
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("ONNX001", result.stdout)

    def test_explain_by_code(self):
        result = self._run("explain", "ONNX001")
        self.assertEqual(result.returncode, 0)
        self.assertIn("ONNX001", result.stdout)
        self.assertIn("empty-graph-name", result.stdout)

    def test_explain_by_name(self):
        result = self._run("explain", "empty-graph-name")
        self.assertEqual(result.returncode, 0)
        self.assertIn("ONNX001", result.stdout)

    def test_explain_unknown_rule(self):
        result = self._run("explain", "NONEXISTENT")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unknown rule", result.stderr)

    def test_check_nonexistent_file(self):
        result = self._run("check", "/tmp/nonexistent_model.onnx")
        self.assertNotEqual(result.returncode, 0)

    def test_check_multiple_nonexistent_files(self):
        result = self._run(
            "check", "/tmp/nonexistent_a.onnx", "/tmp/nonexistent_b.onnx"
        )
        self.assertNotEqual(result.returncode, 0)

    def test_check_empty_directory(self):
        import tempfile  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmpdir:
            result = self._run("check", tmpdir)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("No supported model files found", result.stderr)

    def test_check_help(self):
        result = self._run("check", "--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("--select", result.stdout)
        self.assertIn("--ignore", result.stdout)
        self.assertIn("--output-format", result.stdout)
        self.assertIn("--verbose", result.stdout)

    def test_check_verbose_nonexistent_file(self):
        result = self._run("check", "--verbose", "/tmp/nonexistent_model.onnx")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Checking", result.stderr)
        self.assertIn("nonexistent_model.onnx", result.stderr)


if __name__ == "__main__":
    unittest.main()
