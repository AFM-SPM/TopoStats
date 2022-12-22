"""Fixtures for tracing."""
import pytest

from topostats.tracing.tracedna import traceDNA


@pytest.fixture
def tracedna():
    """Instantiate a class."""
    trace_dna = traceDNA(grain=None, filename="test", pixel_to_nm_scaling=2, skeletonisation_method="zhang")
    return trace_dna
