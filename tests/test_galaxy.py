import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backends import SimulationBackend
from process_gas import Galaxy


class MockBackend(SimulationBackend):
    """Minimal mock backend that doesn't require any simulation data files."""

    def __init__(self, gas_data, star_data):
        self._gas_data = gas_data
        self._star_data = star_data

    def load_gas(self, snap, halo_id, galaxy_id=None):
        return {k: v.copy() if isinstance(v, np.ndarray) else v
                for k, v in self._gas_data.items()}

    def load_stars(self, snap, halo_id, galaxy_id=None):
        return {k: v.copy() if isinstance(v, np.ndarray) else v
                for k, v in self._star_data.items()}

    def load_dm(self, snap, halo_id):
        raise NotImplementedError

    def load_halo_stars(self, snap, halo_id):
        raise NotImplementedError

    def get_redshift(self, snap, halo_row=None):
        return 0.5

    def get_halo_id_column(self):
        return "Halo_id"

    def get_galaxy_id(self, halo_row):
        return int(halo_row.idx.iloc[0])

    def needs_coordinate_offset(self):
        return False

    def needs_velocity_scaling(self):
        return False

    def needs_density_conversion(self):
        return False

    def needs_temperature_computation(self):
        return False

    def needs_hsml_computation(self):
        return False

    def has_virial_radius(self):
        return True

    def has_sfr_dist(self):
        return False

    def has_wind_particles(self):
        return False


@pytest.fixture
def rng():
    return np.random.default_rng(7)


@pytest.fixture
def concentrated_gas(rng):
    """Gas with bimodal velocity structure: galaxy + outflow components.
    Coordinates placed well away from origin to avoid direction=0 issues.
    All within cut_r=30 kpc.
    """
    n_gal, n_out = 150, 50
    n = n_gal + n_out
    coords = np.vstack([
        rng.normal(5, 2, (n_gal, 3)),
        rng.normal(8, 2, (n_out, 3)),
    ]).astype(np.float32)
    velocities = np.vstack([
        rng.normal(0, 80, (n_gal, 3)),
        rng.normal([300, 0, 0], 50, (n_out, 3)),
    ]).astype(np.float32)
    return {
        "Coordinates": coords,
        "Velocities": velocities,
        "Masses": rng.uniform(0.001, 0.01, n).astype(np.float32),
        "Density": rng.uniform(1e-4, 1e-1, n).astype(np.float32),
        "StarFormationRate": np.maximum(rng.normal(0, 0.1, n), 0).astype(np.float32),
        "GFM_Metallicity": rng.uniform(0.001, 0.03, n).astype(np.float32),
        "Temperature": np.full(n, 2e4, dtype=np.float32),
        "count": n,
    }


@pytest.fixture
def concentrated_stars(rng):
    """Stars concentrated near origin so ang_mom_dir uses enough particles
    (needs Relative_Distances < 2 when has_sfr_dist=False).
    """
    n = 80
    return {
        "Coordinates": rng.normal(0, 0.5, (n, 3)).astype(np.float32),
        "Velocities": rng.normal(0, 80, (n, 3)).astype(np.float32),
        "Masses": rng.uniform(0.001, 0.01, n).astype(np.float32),
        "count": n,
    }


@pytest.fixture
def mock_df():
    return pd.DataFrame({
        "Halo_id": [0],
        "snap": [17],
        "idx": [42],
        "R_vir": [150.0],
        "M_star_log": [9.5],
        "r_SFR": [3.0],
        "Galaxy_pos_x": [0.0],
        "Galaxy_pos_y": [0.0],
        "Galaxy_pos_z": [0.0],
        "Galaxy_vel_x": [0.0],
        "Galaxy_vel_y": [0.0],
        "Galaxy_vel_z": [0.0],
        "SubhaloVelDisp": [100.0],
    })


@pytest.fixture
def galaxy(concentrated_gas, concentrated_stars, mock_df):
    backend = MockBackend(concentrated_gas, concentrated_stars)
    return Galaxy(
        df=mock_df,
        halo_id=0,
        snap=17,
        aperture_size=0.6,
        backend=backend,
    )


class TestGalaxyInit:
    def test_halo_lookup(self, galaxy):
        assert len(galaxy.halo) == 1
        assert int(galaxy.halo.Halo_id.iloc[0]) == 0

    def test_gal_pos_from_df(self, galaxy):
        np.testing.assert_array_equal(galaxy.gal_pos, [0.0, 0.0, 0.0])

    def test_gal_vel_from_df(self, galaxy):
        np.testing.assert_array_equal(galaxy.gal_vel, [0.0, 0.0, 0.0])

    def test_critical_velocity(self, galaxy):
        assert galaxy.critical_velocity == pytest.approx(200.0)

    def test_r_vir_loaded(self, galaxy):
        assert galaxy.r_vir == pytest.approx(150.0)

    def test_galaxy_id_from_df(self, galaxy):
        assert galaxy.galaxy_id == 42

    def test_redshift_from_backend(self, galaxy):
        assert galaxy.z == pytest.approx(0.5)

    def test_halo_id_column_used(self, concentrated_gas, concentrated_stars, mock_df):
        """Galaxy should also work when id_col is 'idx' (Serra-style)."""
        class SerraLikeBackend(MockBackend):
            def get_halo_id_column(self):
                return "idx"

        backend = SerraLikeBackend(concentrated_gas, concentrated_stars)
        gal = Galaxy(df=mock_df, halo_id=42, snap=17, aperture_size=0.6, backend=backend)
        assert len(gal.halo) == 1


class TestGalaxyStaticMethods:
    def test_compute_temperature_adds_key(self):
        gas = {
            "InternalEnergy": np.full(10, 1e4, dtype=np.float32),
            "ElectronAbundance": np.full(10, 1.0, dtype=np.float32),
        }
        Galaxy._compute_gas_temperature(gas)
        assert "Temperature" in gas

    def test_compute_temperature_shape(self):
        n = 15
        gas = {
            "InternalEnergy": np.full(n, 5e4, dtype=np.float32),
            "ElectronAbundance": np.full(n, 0.8, dtype=np.float32),
        }
        Galaxy._compute_gas_temperature(gas)
        assert gas["Temperature"].shape == (n,)

    def test_compute_temperature_monotone_with_energy(self):
        low = {"InternalEnergy": np.array([1e3]), "ElectronAbundance": np.array([1.0])}
        high = {"InternalEnergy": np.array([1e5]), "ElectronAbundance": np.array([1.0])}
        Galaxy._compute_gas_temperature(low)
        Galaxy._compute_gas_temperature(high)
        assert high["Temperature"][0] > low["Temperature"][0]

    def test_compute_temperature_positive(self):
        gas = {
            "InternalEnergy": np.array([1e4, 2e4, 5e4], dtype=np.float32),
            "ElectronAbundance": np.array([0.8, 1.0, 1.2], dtype=np.float32),
        }
        Galaxy._compute_gas_temperature(gas)
        assert np.all(gas["Temperature"] > 0)


class TestGalaxyGasProperties:
    def test_gas_has_flow_velocities(self, galaxy):
        assert "Flow_Velocities" in galaxy.gas

    def test_gas_has_relative_velocities(self, galaxy):
        assert "Relative_Velocities" in galaxy.gas

    def test_gas_has_indices(self, galaxy):
        assert "idces" in galaxy.gas

    def test_gas_count_positive(self, galaxy):
        assert galaxy.gas["count"] > 0

    def test_flow_velocities_shape(self, galaxy):
        gas = galaxy.gas
        assert gas["Flow_Velocities"].shape == (gas["count"],)

    def test_no_coordinate_offset_applied(self, galaxy):
        """With needs_coordinate_offset=False, Physical_Coordinates == Coordinates."""
        gas = galaxy.gas
        np.testing.assert_array_equal(
            gas["Physical_Coordinates"], gas["Abs_Coordinates"]
        )


class TestGalaxyOutflowMethods:
    def test_out_gas_is_dict(self, galaxy):
        assert isinstance(galaxy.out_gas, dict)
        assert "count" in galaxy.out_gas

    def test_remain_gas_disjoint_from_out_gas(self, galaxy):
        out_ids = set(galaxy.out_gas["idces"].tolist())
        remain_ids = set(galaxy.remain_gas["idces"].tolist())
        assert len(out_ids & remain_ids) == 0

    def test_out_and_remain_cover_gas(self, galaxy):
        out_ids = set(galaxy.out_gas["idces"].tolist())
        remain_ids = set(galaxy.remain_gas["idces"].tolist())
        all_ids = set(galaxy.gas["idces"].tolist())
        assert out_ids | remain_ids == all_ids

    def test_get_outflow_mass_nonnegative(self, galaxy):
        mass = galaxy.get_outflow_mass()
        assert mass >= 0

    def test_get_outflow_mass_in_aperture(self, galaxy):
        mass_full = galaxy.get_outflow_mass(in_aperture=False)
        mass_ap = galaxy.get_outflow_mass(in_aperture=True)
        assert mass_ap <= mass_full

    def test_get_quantile_velocity_returns_value(self, galaxy):
        if galaxy.out_gas["count"] < 3:
            pytest.skip("too few outflow particles")
        v = galaxy.get_quantile_velocity(0.5, weighting="Masses")
        assert v is None or v >= 0

    def test_get_quantile_velocity_80_geq_20(self, galaxy):
        if galaxy.out_gas["count"] < 3:
            pytest.skip("too few outflow particles")
        v20 = galaxy.get_quantile_velocity(0.2, weighting="Masses")
        v80 = galaxy.get_quantile_velocity(0.8, weighting="Masses")
        if v20 is not None and v80 is not None:
            assert v80 >= v20

    def test_get_flow_rate_finite_or_nan(self, galaxy):
        rate = galaxy.get_flow_rate()
        assert np.isfinite(rate) or np.isnan(rate)

    def test_get_in_aperture_subset(self, galaxy):
        ap = galaxy.get_in_aperture(galaxy.gas)
        assert ap["count"] <= galaxy.gas["count"]
