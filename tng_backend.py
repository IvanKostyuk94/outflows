"""TNG simulation backend using illustris_python."""

import os
import h5py
import numpy as np
import illustris_python as il

from backends import SimulationBackend
from utils import map_to_new_dict

# Snapshot-to-redshift mapping for TNG50 (snaps 0-25)
_TNG_REDSHIFTS = {
    0: 20.046490988807516,
    1: 14.989173240042412,
    2: 11.980213315300293,
    3: 10.975643294137885,
    4: 9.996590466186333,
    5: 9.388771271940549,
    6: 9.00233985416247,
    7: 8.449476294368743,
    8: 8.012172948865935,
    9: 7.5951071498715965,
    10: 7.236276066167360,
    11: 7.005417045544533,
    12: 6.491597745667503,
    13: 6.010757398844900,
    14: 5.846613747881867,
    15: 5.529765807949103,
    16: 5.227580973127337,
    17: 4.995933468164624,
    18: 4.664517702470927,
    19: 4.428033736605549,
    20: 4.176834914726472,
    21: 4.007945111465268,
    22: 3.708774264642235,
    23: 3.490861369260649,
    24: 3.283033057956525,
    25: 3.008131071630377,
}


class TNGBackend(SimulationBackend):
    """Backend for IllustrisTNG simulations."""

    def __init__(self, sim_path=None, config=None):
        if sim_path is not None:
            self.sim_path = sim_path
        elif config is not None:
            self.sim_path = os.path.join(
                config["tng_datapath"], config["sim_name"], "output"
            )
        else:
            raise ValueError("Provide either sim_path or config dict")

    def load_gas(self, snap, halo_id, galaxy_id=None):
        return il.snapshot.loadHalo(self.sim_path, snap, halo_id, "gas")

    def load_stars(self, snap, halo_id, galaxy_id=None):
        if galaxy_id is None:
            raise ValueError("galaxy_id required for TNG star loading")
        stars_and_wind = il.snapshot.loadSubhalo(
            self.sim_path, snap, galaxy_id, "stars"
        )
        relevant = stars_and_wind["GFM_StellarFormationTime"] > 0
        return map_to_new_dict(stars_and_wind, relevant)

    def load_dm(self, snap, halo_id):
        return il.snapshot.loadHalo(self.sim_path, snap, halo_id, "dm")

    def load_halo_stars(self, snap, halo_id):
        return il.snapshot.loadHalo(self.sim_path, snap, halo_id, "stars")

    def get_redshift(self, snap, halo_row=None):
        return _TNG_REDSHIFTS[snap]

    def get_halo_id_column(self):
        return "Halo_id"

    def get_galaxy_id(self, halo_row):
        return int(halo_row.idx.iloc[0])

    def needs_coordinate_offset(self):
        return True

    def needs_velocity_scaling(self):
        return True

    def needs_density_conversion(self):
        return True

    def needs_temperature_computation(self):
        return True

    def needs_hsml_computation(self):
        return True

    def has_virial_radius(self):
        return True

    def has_sfr_dist(self):
        return True

    def has_wind_particles(self):
        return True

    def get_dm_mass(self, snap):
        """Get dark matter particle mass from snapshot header."""
        snap_dir = f"snapdir_{snap:03d}"
        snap_name = f"snap_{snap:03d}"
        snap_path = os.path.join(self.sim_path, snap_dir, snap_name + ".0.hdf5")
        with h5py.File(snap_path, "r") as f:
            dm_mass = dict(f["Header"].attrs.items())["MassTable"][1]
        return dm_mass
