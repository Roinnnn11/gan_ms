import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.metrics.mindspore_backend import metrics_from_features_and_logits
from src.metrics.torch_fidelity_backend import evaluate_with_torch_fidelity


class TorchFidelityBackendTests(unittest.TestCase):
    def test_backend_normalizes_metric_names(self) -> None:
        calls = []

        def calculate_metrics(**kwargs):
            calls.append(kwargs)
            return {
                "frechet_inception_distance": 12.5,
                "inception_score_mean": 2.25,
                "inception_score_std": 0.15,
            }

        fake_module = types.SimpleNamespace(calculate_metrics=calculate_metrics)
        with tempfile.TemporaryDirectory() as root:
            real_dir = Path(root) / "real"
            fake_dir = Path(root) / "fake"
            real_dir.mkdir()
            fake_dir.mkdir()
            (real_dir / "real.png").write_bytes(b"not decoded by mocked backend")
            (fake_dir / "fake.png").write_bytes(b"not decoded by mocked backend")

            with patch.dict(sys.modules, {"torch_fidelity": fake_module}):
                result = evaluate_with_torch_fidelity(
                    real_dir,
                    fake_dir,
                    batch_size=32,
                    cuda=False,
                    splits=5,
                    seed=7,
                )

        self.assertEqual(result["backend"], "torch-fidelity")
        self.assertEqual(result["fid"], 12.5)
        self.assertEqual(result["is_mean"], 2.25)
        self.assertEqual(result["is_std"], 0.15)
        self.assertEqual(calls[0]["batch_size"], 32)
        self.assertEqual(calls[0]["isc_splits"], 5)
        self.assertEqual(calls[0]["rng_seed"], 7)

    def test_backend_rejects_missing_directory_before_import(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "image directory"):
            evaluate_with_torch_fidelity("missing-real", "missing-fake")


class MindSporeBackendTests(unittest.TestCase):
    def test_shared_formula_adapter_returns_normalized_metrics(self) -> None:
        features = np.array(
            [[0.0, 1.0], [1.0, 2.0], [2.0, 0.0], [3.0, 1.0]],
            dtype=np.float64,
        )
        logits = np.zeros((4, 3), dtype=np.float64)

        result = metrics_from_features_and_logits(
            features,
            features.copy(),
            logits,
            splits=2,
            shuffle=False,
        )

        self.assertEqual(result["backend"], "mindspore-mindcv")
        self.assertAlmostEqual(result["fid"], 0.0, places=8)
        self.assertAlmostEqual(result["is_mean"], 1.0, places=8)
        self.assertAlmostEqual(result["is_std"], 0.0, places=8)


if __name__ == "__main__":
    unittest.main()
