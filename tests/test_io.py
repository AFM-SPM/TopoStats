"""Tests of IO"""
from pathlib import Path
from topostats.io import read_yaml

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / 'tests' / 'resources'


def test_read_yaml() -> None:
    """Test reading of YAML file."""
    sample_config = read_yaml(RESOURCES / 'sample_config.yaml')

    assert isinstance(sample_config, dict)
    assert len(sample_config) == 8
    assert isinstance(sample_config['output_dir'], str)
    assert sample_config['output_dir'] == './output'
    assert isinstance(sample_config['cores'], int)
    assert sample_config['cores'] == 4
    assert isinstance(sample_config['quiet'], bool)
    assert sample_config['quiet'] == False
    assert isinstance(sample_config['file_ext'], str)
    assert sample_config['file_ext'] == '.spm'
    assert isinstance(sample_config['channel'], str)
    assert sample_config['channel'] == 'Height'
    assert isinstance(sample_config['grains'], dict)
    assert isinstance(sample_config['grains']['gaussian_size'], float)
    assert sample_config['grains']['gaussian_size'] == 2.0
    assert isinstance(sample_config['grains']['dx'], float)
    assert sample_config['grains']['dx'] == 1.0
    assert isinstance(sample_config['grains']['upper_height_threshold_rms_multiplier'], float)
    assert sample_config['grains']['upper_height_threshold_rms_multiplier'] == 1.0
    assert isinstance(sample_config['grains']['threshold_multiplier'], float)
    assert sample_config['grains']['threshold_multiplier'] == 1.7
    assert isinstance(sample_config['grains']['minimum_grain_size'], int)
    assert sample_config['grains']['minimum_grain_size'] == 800
    assert isinstance(sample_config['grains']['mode'], str)
    assert sample_config['grains']['mode'] == 'nearest'
    assert isinstance(sample_config['grains']['background'], float)
    assert sample_config['grains']['background'] == 0.0
