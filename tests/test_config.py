import os
import sys
import pytest

# Ensure the project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_config_loads_defaults():
    """Config should return sensible defaults even with no paths set."""
    from config import get_config
    cfg = get_config()
    assert cfg["hdf_key"] == "galaxies"
    assert cfg["hdf_ending"] == ".hdf5"
    assert cfg["dir_prefix"] == "snap_"


def test_env_vars_override_yaml(monkeypatch, tmp_path):
    """Environment variables take precedence over YAML values."""
    monkeypatch.setenv("OUTFLOWS_BASE_PATH", "/tmp/test_base")
    monkeypatch.setenv("TNG_DATAPATH", "/tmp/test_tng")
    monkeypatch.setenv("OUTFLOWS_SIM_NAME", "TestSim")

    # Reload config with env vars set
    import importlib
    import config as cfg_module
    importlib.reload(cfg_module)

    cfg = cfg_module.get_config()
    assert cfg["base_path"] == "/tmp/test_base"
    assert cfg["tng_datapath"] == "/tmp/test_tng"
    assert cfg["sim_name"] == "TestSim"

    # Cleanup
    importlib.reload(cfg_module)


def test_config_dict_exported():
    """The module-level 'config' dict should be accessible."""
    from config import config
    assert isinstance(config, dict)
    assert "hdf_key" in config
