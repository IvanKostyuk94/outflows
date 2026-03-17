import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gaussian_outflow_selection import (
    group_gas,
    select_galaxy_group,
    get_only_outflowing_gas,
    normalize,
    get_data,
)
from utils import map_to_new_dict


def make_bimodal_gas(n_galaxy=150, n_outflow=50, rng=None):
    """Create synthetic gas with two clear components:
    - Galaxy component: slow velocities, small distances
    - Outflow component: fast velocities, large distances
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Galaxy component: slow, near
    gal_vel = rng.normal(0, 50, n_galaxy).astype(np.float32)
    gal_dist = rng.uniform(0, 5, n_galaxy).astype(np.float32)
    gal_sfr = rng.uniform(0, 1, n_galaxy).astype(np.float32)
    gal_coords = rng.normal(0, 3, (n_galaxy, 3)).astype(np.float32)

    # Outflow component: fast, far
    out_vel = rng.normal(300, 50, n_outflow).astype(np.float32)
    out_dist = rng.uniform(10, 20, n_outflow).astype(np.float32)
    out_sfr = rng.uniform(0, 0.01, n_outflow).astype(np.float32)
    out_coords = rng.normal(15, 3, (n_outflow, 3)).astype(np.float32)

    n = n_galaxy + n_outflow
    gas = {
        "Flow_Velocities": np.concatenate([gal_vel, out_vel]),
        "Relative_Distances": np.concatenate([gal_dist, out_dist]),
        "StarFormationRate": np.concatenate([gal_sfr, out_sfr]),
        "Coordinates": np.vstack([gal_coords, out_coords]),
        "Masses": rng.uniform(0.001, 0.01, n).astype(np.float32),
        "count": n,
    }
    return gas, n_galaxy


class TestGroupGas:
    def test_finds_two_groups(self):
        gas, _ = make_bimodal_gas()
        group_gas(gas, n_peak=2)
        assert "group" in gas
        assert set(np.unique(gas["group"])) == {0, 1}

    def test_assigns_all_particles(self):
        gas, _ = make_bimodal_gas()
        n = gas["count"]
        group_gas(gas, n_peak=2)
        assert len(gas["group"]) == n

    def test_mass_weighted_path(self):
        gas, _ = make_bimodal_gas()
        group_gas(gas, n_peak=2, mass_weighted=True)
        assert "group" in gas


class TestSelectGalaxyGroup:
    def test_selects_closer_group(self):
        """Group with smaller median distance should be selected."""
        rng = np.random.default_rng(0)
        group_near = {
            "Flow_Velocities": rng.normal(0, 30, 100).astype(np.float32),
            "Relative_Distances": rng.uniform(0, 3, 100).astype(np.float32),
            "Masses": rng.uniform(0.001, 0.01, 100).astype(np.float32),
            "count": 100,
        }
        group_far = {
            "Flow_Velocities": rng.normal(300, 30, 100).astype(np.float32),
            "Relative_Distances": rng.uniform(15, 25, 100).astype(np.float32),
            "Masses": rng.uniform(0.001, 0.01, 100).astype(np.float32),
            "count": 100,
        }
        group_num = select_galaxy_group([group_near, group_far])
        assert group_num == 0

    def test_weighted_distance_mode(self):
        rng = np.random.default_rng(1)
        group_near = {
            "Relative_Distances": rng.uniform(0, 3, 50).astype(np.float32),
            "Masses": rng.uniform(0.001, 0.01, 50).astype(np.float32),
            "count": 50,
        }
        group_far = {
            "Relative_Distances": rng.uniform(15, 25, 50).astype(np.float32),
            "Masses": rng.uniform(0.001, 0.01, 50).astype(np.float32),
            "count": 50,
        }
        group_num = select_galaxy_group(
            [group_near, group_far], use_weighted_distance=True
        )
        assert group_num == 0


class TestGetOnlyOutflowingGas:
    def test_removes_galaxy_group_below_crit(self):
        rng = np.random.default_rng(42)
        n = 100
        gas = {
            "group": np.array([0] * 70 + [1] * 30),
            "Flow_Velocities": np.concatenate([
                rng.normal(50, 20, 70),  # galaxy group
                rng.normal(350, 30, 30),  # outflow group
            ]).astype(np.float32),
            "Masses": rng.uniform(0.001, 0.01, n).astype(np.float32),
            "count": n,
        }
        result = get_only_outflowing_gas(gas, galaxy_group=0, crit_vout=200)
        # All outflow group particles should be in result
        assert result["count"] > 0
        # Particles in galaxy group with slow velocity should be removed
        slow_galaxy = (gas["group"] == 0) & (gas["Flow_Velocities"] < 200)
        assert result["count"] == (~slow_galaxy).sum()


class TestNormalize:
    def test_flow_velocities_linear(self):
        data = np.array([-100.0, 0.0, 100.0, 200.0])
        result = normalize(data, "Flow_Velocities")
        assert result.min() >= 0
        assert result.max() <= 1

    def test_temperature_log(self):
        data = np.array([1e3, 1e4, 1e5, 1e6])
        result = normalize(data, "Temperature")
        assert result.min() >= 0
        assert result.max() <= 1

    def test_unknown_key_raises(self):
        with pytest.raises(NotImplementedError):
            normalize(np.array([1.0, 2.0]), "UnknownKey")
