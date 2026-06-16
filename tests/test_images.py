import numpy as np

from src.utils.images import make_image_grid, tensor_to_uint8_images


def test_tensor_to_uint8_images_converts_nchw_range() -> None:
    images = np.array(
        [
            [[[-1.0, 0.0], [1.0, 2.0]], [[-1.0, 0.0], [1.0, 2.0]], [[-1.0, 0.0], [1.0, 2.0]]]
        ],
        dtype=np.float32,
    )

    converted = tensor_to_uint8_images(images)

    assert converted.shape == (1, 2, 2, 3)
    assert converted[0, 0, 0, 0] == 0
    assert converted[0, 0, 1, 0] == 127
    assert converted[0, 1, 0, 0] == 255
    assert converted[0, 1, 1, 0] == 255


def test_make_image_grid_uses_square_layout_by_default() -> None:
    images = np.zeros((5, 3, 4, 4), dtype=np.float32)

    grid = make_image_grid(images)

    assert grid.shape == (8, 12, 3)
