import os
import h5py
from pyTNG import data_interface as _data_interface
from config import config
from astropy import units as u
from astropy.constants import G
from pyTNG.cosmology import TNGcosmo


def get_sim():
    basepath = "/virgotng/universe/IllustrisTNG/"
    sim_name = "L35n2160TNG"
    sim = _data_interface.TNG50Simulation(os.path.join(basepath, sim_name))
    sim_path = os.path.join(basepath, sim_name, "output")
    return sim, sim_path


def get_redshift(snap_num):
    sim, _ = get_sim()
    z = sim.snap_cat[snap_num].header["Redshift"]
    return z


def scale_factor(z):
    return 1 / (z + 1)


def get_snap_name(snap_num):
    if snap_num < 10:
        return f"snap_00{snap_num}"
    else:
        return f"snap_0{snap_num}"


def get_snap_dir(snap_num):
    if snap_num < 10:
        return f"snapdir_00{snap_num}"
    else:
        return f"snapdir_0{snap_num}"


def get_mass_in_kg(mass):
    msun_to_kg = (1 * u.M_sun).to(u.kg)
    mass = mass * 1e10 / TNGcosmo.h * msun_to_kg
    return mass.value


def get_dist_in_km(dist, z):
    kpc_to_km = (1 * u.kpc).to(u.km)
    dist = dist / (z + 1) / TNGcosmo.h * kpc_to_km
    return dist.value


# get dm particle mass in 1e10/h M_sun
def get_dm_mass(snap):
    _, sim_path = get_sim()
    snap_dir = get_snap_dir(snap)
    snap_name = get_snap_name(snap)
    snap_path = os.path.join(sim_path, snap_dir, snap_name + ".0.hdf5")
    snap_file = h5py.File(snap_path)
    dm_mass = dict(snap_file["Header"].attrs.items())["MassTable"][1]
    snap_file.close()
    return dm_mass
