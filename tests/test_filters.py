import numpy as np
import pytest

import topostats.filters as filters


@pytest.fixture
def random_image() -> np.array:
    rng = np.random.default_rng(seed=1000)
    return rng.random((1024, 1024))


def test_amplify(random_image: np.array):
    filtered = filters.amplify(random_image, 1.5)

    target = random_image * 1.5

    np.testing.assert_array_equal(filtered, target)
