import os
import sys
import numpy as np
import pytest

# Mock illustris_python before importing tng_backend (not installed in test env)
from unittest.mock import MagicMock, patch
sys.modules.setdefault("illustris_python", MagicMock())
sys.modules.setdefault("illustris_python.snapshot", MagicMock())

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tng_backend import TNGBackend, _TNG_REDSHIFTS
from serra_backend import SerraBackend, _transform_gas, _transform_stars


# ---------------------------------------------------------------------------
# TNGBackend
# ---------------------------------------------------------------------------

class TestTNGBackendFlags:
    """Flag methods should return hard-coded values; no real data needed."""

    @pytest.fixture(autouse=True)
    def backend(self):
        self.b = TNGBackend(sim_path="/fake/path")

    def test_needs_coordinate_offset(self):
        assert self.b.needs_coordinate_offset() is True

    def test_needs_velocity_scaling(self):
        assert self.b.needs_velocity_scaling() is True

    def test_needs_density_conversion(self):
        assert self.b.needs_density_conversion() is True

    def test_needs_temperature_computation(self):
        assert self.b.needs_temperature_computation() is True

    def test_needs_hsml_computation(self):
        assert self.b.needs_hsml_computation() is True

    def test_has_virial_radius(self):
        assert self.b.has_virial_radius() is True

    def test_has_sfr_dist(self):
        assert self.b.has_sfr_dist() is True

    def test_has_wind_particles(self):
        assert self.b.has_wind_particles() is True

    def test_gmm_not_mass_weighted(self):
        assert self.b.gmm_mass_weighted() is False

    def test_gmm_not_distance_weighted(self):
        assert self.b.gmm_distance_weighted() is False

    def test_halo_id_column(self):
        assert self.b.get_halo_id_column() == "Halo_id"

    def test_mean_velocity_weights_none(self):
        assert self.b.get_mean_velocity_weights({}) is None


class TestTNGBackendRedshifts:
    @pytest.fixture(autouse=True)
    def backend(self):
        self.b = TNGBackend(sim_path="/fake/path")

    def test_snap_0_high_redshift(self):
        assert self.b.get_redshift(0) > 15

    def test_snap_25_low_redshift(self):
        z = self.b.get_redshift(25)
        assert 0 < z < 5

    def test_snap_17_approx(self):
        z = self.b.get_redshift(17)
        assert abs(z - 4.996) < 0.01

    def test_redshifts_monotone_decreasing(self):
        snaps = sorted(_TNG_REDSHIFTS.keys())
        zs = [_TNG_REDSHIFTS[s] for s in snaps]
        assert all(zs[i] > zs[i + 1] for i in range(len(zs) - 1))

    def test_invalid_snap_raises(self):
        with pytest.raises(KeyError):
            self.b.get_redshift(999)

    def test_all_redshifts_positive(self):
        assert all(z > 0 for z in _TNG_REDSHIFTS.values())


class TestTNGBackendConstructor:
    def test_sim_path_direct(self):
        b = TNGBackend(sim_path="/data/tng")
        assert b.sim_path == "/data/tng"

    def test_config_constructs_path(self):
        cfg = {"tng_datapath": "/data", "sim_name": "TNG50-1"}
        b = TNGBackend(config=cfg)
        assert "TNG50-1" in b.sim_path
        assert "output" in b.sim_path

    def test_no_args_raises(self):
        with pytest.raises(ValueError):
            TNGBackend()


# ---------------------------------------------------------------------------
# SerraBackend
# ---------------------------------------------------------------------------

class TestSerraBackendFlags:
    @pytest.fixture(autouse=True)
    def backend(self):
        self.b = SerraBackend(base_path="/fake")

    def test_needs_no_coordinate_offset(self):
        assert self.b.needs_coordinate_offset() is False

    def test_needs_no_velocity_scaling(self):
        assert self.b.needs_velocity_scaling() is False

    def test_needs_no_density_conversion(self):
        assert self.b.needs_density_conversion() is False

    def test_needs_no_temperature_computation(self):
        assert self.b.needs_temperature_computation() is False

    def test_needs_no_hsml_computation(self):
        assert self.b.needs_hsml_computation() is False

    def test_has_no_virial_radius(self):
        assert self.b.has_virial_radius() is False

    def test_has_no_sfr_dist(self):
        assert self.b.has_sfr_dist() is False

    def test_has_no_wind_particles(self):
        assert self.b.has_wind_particles() is False

    def test_gmm_mass_weighted(self):
        assert self.b.gmm_mass_weighted() is True

    def test_gmm_distance_weighted(self):
        assert self.b.gmm_distance_weighted() is True

    def test_halo_id_column(self):
        assert self.b.get_halo_id_column() == "idx"


def _make_raw_gas(n=50, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "pos": rng.normal(0, 5, (n, 3)).astype(np.float32),
        "vel": rng.normal(0, 100, (n, 3)).astype(np.float32),
        "mass": rng.uniform(1e-4, 1e-3, n).astype(np.float32),
        "temp": rng.uniform(1e3, 1e6, n).astype(np.float32),
        "smooth": rng.uniform(0.1, 1.0, n).astype(np.float32),
        "metal": rng.uniform(0.001, 0.03, n).astype(np.float32),
        "rho": rng.uniform(1e-4, 1e-1, n).astype(np.float32),
        "H": rng.uniform(0.5, 0.8, n).astype(np.float32),
        "H+": rng.uniform(0.0, 0.3, n).astype(np.float32),
        "H2": rng.uniform(0.0, 0.1, n).astype(np.float32),
        "H2+": np.zeros(n, dtype=np.float32),
        "HE": rng.uniform(0.05, 0.1, n).astype(np.float32),
        "HE+": np.zeros(n, dtype=np.float32),
        "HE++": np.zeros(n, dtype=np.float32),
        "z": 0.5,
    }


def _make_raw_stars(n=30, seed=1):
    rng = np.random.default_rng(seed)
    return {
        "pos": rng.normal(0, 5, (n, 3)).astype(np.float32),
        "vel": rng.normal(0, 80, (n, 3)).astype(np.float32),
        "mass": rng.uniform(1e-4, 1e-3, n).astype(np.float32),
        "z": 0.5,
    }


class TestSerraTransformGas:
    def test_standardizes_keys(self):
        result = _transform_gas(_make_raw_gas())
        for key in ("Coordinates", "Velocities", "Masses", "Temperature",
                    "StarFormationRate", "GFM_Metallicity", "Density"):
            assert key in result, f"Missing key: {key}"

    def test_removes_z_key(self):
        result = _transform_gas(_make_raw_gas())
        assert "z" not in result

    def test_count_matches_n(self):
        n = 40
        result = _transform_gas(_make_raw_gas(n=n))
        assert result["count"] == n

    def test_coordinates_shape(self):
        n = 50
        result = _transform_gas(_make_raw_gas(n=n))
        assert result["Coordinates"].shape == (n, 3)

    def test_sfr_nonnegative(self):
        result = _transform_gas(_make_raw_gas())
        assert np.all(result["StarFormationRate"] >= 0)

    def test_hsml_twice_smooth(self):
        raw = _make_raw_gas()
        smooth = raw["smooth"].copy()
        result = _transform_gas(raw)
        np.testing.assert_allclose(result["hsml"], 2 * smooth)


class TestSerraTransformStars:
    def test_standardizes_keys(self):
        result = _transform_stars(_make_raw_stars())
        for key in ("Coordinates", "Velocities", "Masses"):
            assert key in result, f"Missing key: {key}"

    def test_removes_z_key(self):
        result = _transform_stars(_make_raw_stars())
        assert "z" not in result

    def test_count_correct(self):
        n = 25
        result = _transform_stars(_make_raw_stars(n=n))
        assert result["count"] == n

    def test_coordinates_shape(self):
        n = 30
        result = _transform_stars(_make_raw_stars(n=n))
        assert result["Coordinates"].shape == (n, 3)


class TestSerraBackendLoadWithMock:
    """Test load_gas / load_stars by patching _load_pickle."""

    def test_load_gas_returns_standardized_dict(self):
        raw = _make_raw_gas()
        b = SerraBackend(base_path="/fake")
        with patch("serra_backend._load_pickle", return_value=raw):
            result = b.load_gas(snap=5, halo_id=10)
        assert "Coordinates" in result
        assert "Masses" in result
        assert result["count"] == 50

    def test_load_stars_returns_standardized_dict(self):
        raw = _make_raw_stars()
        b = SerraBackend(base_path="/fake")
        with patch("serra_backend._load_pickle", return_value=raw):
            result = b.load_stars(snap=5, halo_id=10)
        assert "Coordinates" in result
        assert result["count"] == 30
