import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils import (
    map_to_new_dict,
    sort_all_keys,
    scale_factor,
    get_redshift,
    get_snap_name,
    get_snap_dir,
    dfFromArrDict,
    calculate_acc,
)


def make_particles(n=10):
    rng = np.random.default_rng(42)
    return {
        "Coordinates": rng.normal(0, 1, (n, 3)).astype(np.float32),
        "Masses": rng.uniform(0.001, 0.01, n).astype(np.float32),
        "Temperature": rng.uniform(1e3, 1e6, n).astype(np.float32),
        "count": n,
    }


class TestMapToNewDict:
    def test_filters_correctly(self):
        p = make_particles(10)
        mask = np.array([True, False] * 5)
        result = map_to_new_dict(p, mask)
        assert result["count"] == 5
        assert len(result["Masses"]) == 5
        assert len(result["Coordinates"]) == 5

    def test_updates_count(self):
        p = make_particles(10)
        mask = np.ones(10, dtype=bool)
        mask[0] = False
        result = map_to_new_dict(p, mask)
        assert result["count"] == 9

    def test_handles_scalar_gracefully(self):
        p = {"Coordinates": np.ones((5, 3)), "scalar_val": 42.0, "count": 5}
        mask = np.ones(5, dtype=bool)
        mask[0] = False
        result = map_to_new_dict(p, mask)
        assert "Coordinates" in result
        assert result["count"] == 4


class TestSortAllKeys:
    def test_sorts_by_key(self):
        p = {
            "Distances": np.array([3.0, 1.0, 2.0]),
            "Masses": np.array([0.3, 0.1, 0.2]),
            "count": 3,
        }
        sort_all_keys(p, "Distances")
        assert list(p["Distances"]) == [1.0, 2.0, 3.0]
        assert list(p["Masses"]) == [0.1, 0.2, 0.3]

    def test_skips_count_key(self):
        p = {
            "Distances": np.array([2.0, 1.0]),
            "count": 2,
        }
        sort_all_keys(p, "Distances")
        assert p["count"] == 2


class TestScaleFactor:
    def test_z_zero(self):
        assert scale_factor(0) == 1.0

    def test_z_positive(self):
        assert abs(scale_factor(1.0) - 0.5) < 1e-10

    def test_z_negative_raises(self):
        # scale factor > 1 for z < 0 (future)
        assert scale_factor(-0.5) > 1.0


class TestGetRedshift:
    def test_snap_0(self):
        z = get_redshift(0)
        assert abs(z - 20.046) < 0.01

    def test_snap_25(self):
        z = get_redshift(25)
        assert abs(z - 3.008) < 0.01

    def test_invalid_snap_raises(self):
        with pytest.raises(KeyError):
            get_redshift(999)


class TestSnapNames:
    def test_snap_name_single_digit(self):
        assert get_snap_name(1) == "snap_001"

    def test_snap_name_double_digit(self):
        assert get_snap_name(15) == "snap_015"

    def test_snap_dir(self):
        assert get_snap_dir(5) == "snapdir_005"


class TestDfFromArrDict:
    def test_1d_arrays(self):
        d = {
            "a": np.array([1, 2, 3]),
            "b": np.array([4, 5, 6]),
        }
        df = dfFromArrDict(d)
        assert ("a", 0) in df.columns
        assert ("b", 0) in df.columns
        assert len(df) == 3

    def test_2d_arrays(self):
        d = {"pos": np.random.randn(5, 3)}
        df = dfFromArrDict(d)
        assert ("pos", 0) in df.columns
        assert ("pos", 1) in df.columns
        assert ("pos", 2) in df.columns

    def test_warns_on_scalar(self):
        d = {"a": np.array([1, 2]), "scalar": 42}
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dfFromArrDict(d)
            assert len(w) == 1
            assert "scalar" in str(w[0].message)

    def test_3d_array_raises(self):
        d = {"bad": np.ones((2, 2, 2))}
        with pytest.raises(RuntimeError):
            dfFromArrDict(d)
