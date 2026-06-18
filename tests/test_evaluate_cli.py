import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.evaluate import run_evaluation


class EvaluateCliTests(unittest.TestCase):
    def test_both_backend_results_are_written_to_json(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            real_dir = root_path / "real"
            fake_dir = root_path / "fake"
            output = root_path / "metrics.json"
            real_dir.mkdir()
            fake_dir.mkdir()
            for index in range(2):
                (real_dir / f"{index}.png").write_bytes(b"real")
                (fake_dir / f"{index}.png").write_bytes(b"fake")

            args = argparse.Namespace(
                real_dir=str(real_dir),
                fake_dir=str(fake_dir),
                backend="both",
                output=str(output),
                batch_size=16,
                splits=2,
                seed=9,
                torch_device="cpu",
                device_target="GPU",
                device_id=0,
            )
            torch_result = {"backend": "torch-fidelity", "fid": 10.0, "is_mean": 2.0, "is_std": 0.1}
            mindspore_result = {
                "backend": "mindspore-mindcv",
                "fid": 11.0,
                "is_mean": 1.9,
                "is_std": 0.2,
            }

            with patch("scripts.evaluate.evaluate_with_torch_fidelity", return_value=torch_result), patch(
                "scripts.evaluate.evaluate_with_mindspore", return_value=mindspore_result
            ):
                payload = run_evaluation(args)

            saved = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(payload, saved)
        self.assertEqual(saved["protocol"]["real_samples"], 2)
        self.assertEqual(saved["protocol"]["fake_samples"], 2)
        self.assertEqual(saved["results"]["torch"]["fid"], 10.0)
        self.assertEqual(saved["results"]["mindspore"]["fid"], 11.0)

    def test_evaluation_rejects_different_sample_counts(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            real_dir = root_path / "real"
            fake_dir = root_path / "fake"
            real_dir.mkdir()
            fake_dir.mkdir()
            (real_dir / "0.png").write_bytes(b"real")
            (real_dir / "1.png").write_bytes(b"real")
            (fake_dir / "0.png").write_bytes(b"fake")
            args = argparse.Namespace(
                real_dir=str(real_dir),
                fake_dir=str(fake_dir),
                backend="torch",
                output=str(root_path / "metrics.json"),
                batch_size=16,
                splits=1,
                seed=9,
                torch_device="cpu",
                device_target="GPU",
                device_id=0,
            )

            with self.assertRaisesRegex(ValueError, "same number"):
                run_evaluation(args)


if __name__ == "__main__":
    unittest.main()
