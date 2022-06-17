"""Tests the dnatracing module"""
from pathlib import Path
import numpy as np
import pytest

import pandas as pd


def test_dnatracing_minicircle(minicircle_dnatracing, minicircle_dnastats) -> None:
    pd.testing.assert_frame_equal(minicircle_dnatracing, minicircle_dnastats)
