import numpy as np
import pytest

import topostats.filters as filters

@pytest.fixture
def random_image() -> np.array:
    return np.random.rand(1024, 1024)

def test_turner(random_image:np.array):
    filtered = filters.turner(random_image, 0)
    target = np.zeros_like(random_image)

    np.testing.assert_array_equal(filtered, target)