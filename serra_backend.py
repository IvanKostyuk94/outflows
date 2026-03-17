"""Serra simulation backend."""

import os
import pickle
import numpy as np
import astropy.units as u
from astropy.constants import G

from backends import SimulationBackend
from utils import map_to_new_dict


def _get_mu(galaxy):
    """Compute mean molecular weight from species abundances."""
    H = galaxy["H"] + galaxy["H+"]
    H2 = galaxy["H2"] + galaxy["H2+"]
    He = galaxy["HE"] + galaxy["HE+"] + galaxy["HE++"]
    return H + 2 * H2 + 4 * He


def _compute_serra_sfr(galaxy):
    """Compute SFR from gas particle properties."""
    t_ff = np.sqrt(3 * np.pi / (32 * G * galaxy["rho"] * u.Msun / u.kpc**3))
    mu = _get_mu(galaxy)
    sfr = (0.1 * galaxy["mass"] * u.M_sun * galaxy["H2"] * mu / t_ff).to("Msun/yr")
    return sfr.value


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _transform_gas(galaxy):
    """Transform Serra raw gas data to the standardized particle dict format."""
    galaxy["Density"] = galaxy["rho"]
    galaxy["Coordinates"] = galaxy["pos"]
    galaxy["Temperature"] = galaxy["temp"]
    galaxy["count"] = len(galaxy["mass"])
    galaxy["Velocities"] = galaxy["vel"]
    galaxy["hsml"] = 2 * galaxy["smooth"]
    galaxy["Masses"] = galaxy["mass"]
    galaxy["StarFormationRate"] = _compute_serra_sfr(galaxy)
    galaxy["GFM_Metallicity"] = galaxy["metal"]
    del galaxy["z"]
    return galaxy


def _transform_stars(galaxy):
    """Transform Serra raw star data to the standardized particle dict format."""
    galaxy["Coordinates"] = galaxy["pos"]
    galaxy["count"] = len(galaxy["mass"])
    galaxy["Velocities"] = galaxy["vel"]
    galaxy["Masses"] = galaxy["mass"]
    del galaxy["z"]
    return galaxy


def _weighted_avg_and_std(values, weights, average):
    """Return weighted standard deviation of velocities."""
    velocities = np.linalg.norm(values, axis=1)
    average_vel = np.linalg.norm(average)
    variance = np.average((velocities - average_vel) ** 2, weights=weights)
    return np.sqrt(variance)


def _get_galaxy_pos_vel(particles):
    """Compute mass-weighted position and velocity of inner particles."""
    keys_to_include = {"pos", "vel", "mass"}
    rel_particles = {k: particles[k] for k in keys_to_include if k in particles}
    rel_particles["rel_pos"] = np.linalg.norm(rel_particles["pos"], axis=1)

    rel_idx = rel_particles["rel_pos"] < 2.5
    rel_particles = map_to_new_dict(rel_particles, rel_idx)
    pos = np.average(rel_particles["pos"], axis=0, weights=rel_particles["mass"])
    vel = np.average(rel_particles["vel"], axis=0, weights=rel_particles["mass"])
    disp = _weighted_avg_and_std(rel_particles["vel"], rel_particles["mass"], vel)
    return pos, vel, disp


class SerraBackend(SimulationBackend):
    """Backend for Serra simulations (pickle-based data)."""

    def __init__(self, base_path=None, config=None):
        if base_path is not None:
            self.base_path = base_path
        elif config is not None:
            self.base_path = config["base_path"]
        else:
            raise ValueError("Provide either base_path or config dict")

    def _gas_path(self, snap, galaxy_id):
        snapdir = "snap" + str(snap)
        return os.path.join(
            self.base_path, "main", "gas", snapdir, str(galaxy_id) + ".pickle"
        )

    def _star_path(self, snap, galaxy_id):
        snapdir = "snap" + str(snap)
        return os.path.join(
            self.base_path, "main", "star", snapdir, str(galaxy_id) + ".pickle"
        )

    def load_gas(self, snap, halo_id, galaxy_id=None):
        gid = galaxy_id if galaxy_id is not None else halo_id
        raw = _load_pickle(self._gas_path(snap, gid))
        return _transform_gas(raw)

    def load_stars(self, snap, halo_id, galaxy_id=None):
        gid = galaxy_id if galaxy_id is not None else halo_id
        raw = _load_pickle(self._star_path(snap, gid))
        return _transform_stars(raw)

    def load_dm(self, snap, halo_id):
        raise NotImplementedError("Serra does not provide separate DM particles")

    def load_halo_stars(self, snap, halo_id):
        raise NotImplementedError("Serra does not provide halo-level star loading")

    def get_redshift(self, snap, halo_row=None):
        if halo_row is not None and "z" in halo_row.columns:
            return halo_row.z.values[0]
        raise ValueError("Serra requires halo_row with 'z' column for redshift")

    def get_halo_id_column(self):
        return "idx"

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
        return False

    def has_sfr_dist(self):
        return False

    def has_wind_particles(self):
        return False

    def get_mean_velocity_weights(self, particles):
        """Serra uses mass-weighted average of inner particles (r < 5 kpc)."""
        mask = particles["Relative_Distances"] < 5
        weights = np.zeros(len(particles["Masses"]))
        weights[mask] = particles["mass"][mask]
        return weights

    def gmm_mass_weighted(self):
        return True

    def gmm_distance_weighted(self):
        return True


def build_serra_df(base_path, name="serra_base", config=None):
    """Build a galaxy DataFrame from Serra pickle files.

    This is a standalone utility, not part of the backend interface.
    """
    from config import config as _cfg

    cfg = config or _cfg
    serra_path = os.path.join(base_path, "data_new")
    serra_files = os.listdir(serra_path)
    df_path = os.path.join(cfg["base_path"], name + cfg["hdf_ending"])

    data_dict = {
        "snap": [], "z": [], "Galaxy_M_star": [], "idx": [],
        "Galaxy_pos_x": [], "Galaxy_pos_y": [], "Galaxy_pos_z": [],
        "Galaxy_vel_x": [], "Galaxy_vel_y": [], "Galaxy_vel_z": [],
        "SubhaloVelDisp": [],
    }

    for snapdir in serra_files:
        snap = int(snapdir[-2:])
        galaxies = os.listdir(os.path.join(serra_path, snapdir))
        for galaxy_id in galaxies:
            file_path = os.path.join(serra_path, snapdir, galaxy_id)
            galaxy = _load_pickle(file_path)
            pos, vel, disp = _get_galaxy_pos_vel(galaxy)
            data_dict["snap"].append(snap)
            data_dict["z"].append(galaxy["z"])
            try:
                data_dict["Galaxy_M_star"].append(galaxy["star_mass"])
            except KeyError:
                continue
            data_dict["idx"].append(int(galaxy_id.split(".")[0]))
            data_dict["Galaxy_pos_x"].append(pos[0])
            data_dict["Galaxy_pos_y"].append(pos[1])
            data_dict["Galaxy_pos_z"].append(pos[2])
            data_dict["Galaxy_vel_x"].append(vel[0])
            data_dict["Galaxy_vel_y"].append(vel[1])
            data_dict["Galaxy_vel_z"].append(vel[2])
            data_dict["SubhaloVelDisp"].append(disp)

    import pandas as pd
    for key in data_dict:
        data_dict[key] = np.array(data_dict[key])
    df = pd.DataFrame(data_dict)
    df["M_star_log"] = np.log10(df["Galaxy_M_star"])
    df.to_hdf(df_path, key=cfg["hdf_key"])
