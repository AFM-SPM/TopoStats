"""Tests the dnatracing module"""
from topostats.tracing.dnatracing import dnaTrace


def test_dnatracing_regression(regtest, minicircle_dnatracing: dnaTrace) -> None:
    """Regression test for dnatracing."""
    print(minicircle_dnatracing.to_string(), file=regtest)
