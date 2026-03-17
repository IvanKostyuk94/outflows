import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def mock_gas_particles(rng):
    """Gas particle dict mimicking TNG snapshot output."""
    n = 200
    return {
        "Coordinates": rng.normal(0, 50, (n, 3)).astype(np.float32),
        "Velocities": rng.normal(0, 100, (n, 3)).astype(np.float32),
        "Masses": rng.uniform(0.001, 0.01, n).astype(np.float32),
        "Density": rng.uniform(1e-4, 1e-1, n).astype(np.float32),
        "InternalEnergy": rng.uniform(1e2, 1e5, n).astype(np.float32),
        "ElectronAbundance": rng.uniform(0.5, 1.2, n).astype(np.float32),
        "StarFormationRate": np.maximum(rng.normal(0, 0.5, n), 0).astype(np.float32),
        "GFM_Metallicity": rng.uniform(0.001, 0.03, n).astype(np.float32),
        "count": n,
    }


@pytest.fixture
def mock_star_particles(rng):
    """Star particle dict mimicking TNG snapshot output."""
    n = 150
    formation_times = rng.uniform(0.1, 1.0, n).astype(np.float32)
    return {
        "Coordinates": rng.normal(0, 30, (n, 3)).astype(np.float32),
        "Velocities": rng.normal(0, 80, (n, 3)).astype(np.float32),
        "Masses": rng.uniform(0.001, 0.01, n).astype(np.float32),
        "GFM_StellarFormationTime": formation_times,
        "count": n,
    }


@pytest.fixture
def mock_halo_df():
    """DataFrame row mimicking the galaxy catalog used by Galaxy class."""
    data = {
        "Halo_id": [0],
        "snap": [17],
        "idx": [42],
        "R_vir": [150.0],
        "M_star_log": [9.5],
        "r_SFR": [3.0],
        "Galaxy_pos_x": [100.0],
        "Galaxy_pos_y": [200.0],
        "Galaxy_pos_z": [300.0],
        "Galaxy_vel_x": [50.0],
        "Galaxy_vel_y": [-30.0],
        "Galaxy_vel_z": [10.0],
        "SubhaloVelDisp": [120.0],
        "Galaxy_SFR": [1.5],
        "Galaxy_M_star": [0.03],
        "Galaxy_M_gas": [0.05],
        "Galaxy_M_wind": [0.001],
        "Galaxy_star_fraction": [0.6],
        "Galaxy_gas_fraction": [0.4],
        "Galaxy_GHMR": [5.0],
        "Galaxy_SHMR": [2.0],
        "Halo_M": [1.0],
        "Halo_M_star": [0.05],
        "Halo_M_gas": [0.1],
    }
    return pd.DataFrame(data)
