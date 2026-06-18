import unittest

import numpy as np

from src.metrics.core import (
    activation_statistics,
    frechet_distance,
    inception_score_from_logits,
)


class MetricCoreTests(unittest.TestCase):
    def test_identical_features_have_zero_fid(self) -> None:
        features = np.array(
            [[0.0, 1.0], [1.0, 2.0], [2.0, 0.0], [3.0, 1.0]],
            dtype=np.float64,
        )
        mean, covariance = activation_statistics(features)

        score = frechet_distance(mean, covariance, mean, covariance)

        self.assertAlmostEqual(score, 0.0, places=8)

    def test_shifted_features_have_positive_fid(self) -> None:
        real = np.array(
            [[0.0, 0.0], [1.0, 0.5], [2.0, 1.0], [3.0, 1.5]],
            dtype=np.float64,
        )
        fake = real + np.array([2.0, -1.0])
        mean_real, covariance_real = activation_statistics(real)
        mean_fake, covariance_fake = activation_statistics(fake)

        score = frechet_distance(mean_real, covariance_real, mean_fake, covariance_fake)

        self.assertAlmostEqual(score, 5.0, places=7)

    def test_uniform_logits_have_inception_score_one(self) -> None:
        logits = np.zeros((100, 10), dtype=np.float64)

        mean, std = inception_score_from_logits(logits, splits=10, shuffle=False)

        self.assertAlmostEqual(mean, 1.0, places=8)
        self.assertAlmostEqual(std, 0.0, places=8)

    def test_confident_diverse_logits_have_higher_inception_score(self) -> None:
        logits = np.full((100, 10), -10.0, dtype=np.float64)
        logits[np.arange(100), np.arange(100) % 10] = 10.0

        mean, _ = inception_score_from_logits(logits, splits=10, shuffle=False)

        self.assertGreater(mean, 9.9)

    def test_statistics_reject_single_sample(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two"):
            activation_statistics(np.ones((1, 3), dtype=np.float64))

    def test_inception_score_rejects_more_splits_than_samples(self) -> None:
        with self.assertRaisesRegex(ValueError, "splits"):
            inception_score_from_logits(np.ones((2, 3)), splits=3)


if __name__ == "__main__":
    unittest.main()
