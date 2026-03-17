import os
import yaml

_DEFAULTS = {
    "hdf_key": "galaxies",
    "dir_prefix": "snap_",
    "df_name": "testing_df_",
    "hdf_ending": ".hdf5",
    "cutout_scale": "1.1",
}

_ENV_MAP = {
    "base_path": "OUTFLOWS_BASE_PATH",
    "tng_datapath": "TNG_DATAPATH",
    "sim_name": "OUTFLOWS_SIM_NAME",
}


def _load_yaml():
    """Load config from YAML file if it exists next to this module."""
    yaml_path = os.path.join(os.path.dirname(__file__), "config_parameters.yml")
    if os.path.exists(yaml_path):
        with open(yaml_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def get_config():
    """Build config: defaults < YAML < environment variables."""
    cfg = dict(_DEFAULTS)
    cfg.update(_load_yaml())

    # Environment variables override everything
    for key, env_var in _ENV_MAP.items():
        val = os.environ.get(env_var)
        if val is not None:
            cfg[key] = val

    # Derive paths if components are available
    if "merger_history_path" not in cfg and "tng_datapath" in cfg and "sim_name" in cfg:
        cfg["merger_history_path"] = os.path.join(
            cfg["tng_datapath"], cfg["sim_name"],
            "postprocessing/trees/SubLink_gal/tree_extended.hdf5",
        )
    if "sfr_hist_path" not in cfg and "tng_datapath" in cfg and "sim_name" in cfg:
        cfg["sfr_hist_path"] = os.path.join(
            cfg["tng_datapath"], cfg["sim_name"],
            "postprocessing/StarFormationRates",
        )

    return cfg


# Backward-compatible global config dict
config = get_config()
