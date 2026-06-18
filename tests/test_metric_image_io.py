import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.metrics.image_io import (
    discover_image_paths,
    prepare_real_images,
    save_generated_images,
)


def _write_image(path: Path, size: tuple[int, int], value: int) -> None:
    array = np.full((size[1], size[0], 3), value, dtype=np.uint8)
    Image.fromarray(array).save(path)


class MetricImageIoTests(unittest.TestCase):
    def test_discovery_is_recursive_and_sorted(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            source = Path(root)
            (source / "nested").mkdir()
            _write_image(source / "b.jpg", (4, 4), 10)
            _write_image(source / "nested" / "a.png", (4, 4), 20)
            (source / "notes.txt").write_text("ignored", encoding="utf-8")

            paths = discover_image_paths(source)

            self.assertEqual([path.name for path in paths], ["b.jpg", "a.png"])

    def test_prepare_real_images_resizes_and_uses_stable_names(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "source"
            output = Path(root) / "output"
            source.mkdir()
            _write_image(source / "000002.jpg", (12, 8), 50)
            _write_image(source / "000001.jpg", (8, 12), 100)

            count = prepare_real_images(source, output, image_size=6, num_images=2)

            self.assertEqual(count, 2)
            self.assertEqual([path.name for path in sorted(output.iterdir())], ["000000.png", "000001.png"])
            with Image.open(output / "000000.png") as image:
                self.assertEqual(image.size, (6, 6))
                self.assertEqual(image.mode, "RGB")

    def test_save_generated_images_converts_minus_one_to_one_range(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            output = Path(root)
            images = np.stack(
                [
                    np.full((3, 2, 2), -1.0, dtype=np.float32),
                    np.full((3, 2, 2), 1.0, dtype=np.float32),
                ]
            )

            next_index = save_generated_images(images, output, start_index=4)

            self.assertEqual(next_index, 6)
            with Image.open(output / "000004.png") as image:
                self.assertEqual(np.asarray(image).max(), 0)
            with Image.open(output / "000005.png") as image:
                self.assertEqual(np.asarray(image).min(), 255)

    def test_prepare_real_images_rejects_nonempty_output(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "source"
            output = Path(root) / "output"
            source.mkdir()
            output.mkdir()
            _write_image(source / "one.jpg", (4, 4), 20)
            (output / "existing.txt").write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "empty"):
                prepare_real_images(source, output, image_size=4, num_images=1)


if __name__ == "__main__":
    unittest.main()
