"""Test validation function."""
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from schema import Or, Schema, SchemaError
import pytest

from topostats.validation import validate_config

TEST_SCHEMA = Schema(
    {
        "a": Path,
        "b": Or("aa", "bb", error="Invalid value in config, valid values are 'aa' or 'bb"),
        "positive_integer": lambda n: 0 < n,
    }
)


@pytest.mark.parametrize(
    "config, expectation",
    [
        # A valid configuration
        ({"a": Path(), "b": "aa", "positive_integer": 4}, does_not_raise()),
        # Invalid value for a (string instead of Path)
        ({"a": "path", "b": "aa", "positive_integer": 4}, pytest.raises(SchemaError)),
        # Invalid value for b (int instead of string)
        ({"a": Path(), "b": 3, "positive_integer": 4}, pytest.raises(SchemaError)),
        # Invalid value for positive_integer (int instead of string)
        ({"a": Path(), "b": 3, "positive_integer": -4}, pytest.raises(SchemaError)),
    ],
)
def test_validate(config, expectation) -> None:
    """Test various configurations."""
    with expectation:
        validate_config(config, schema=TEST_SCHEMA, config_type="Test YAML")
