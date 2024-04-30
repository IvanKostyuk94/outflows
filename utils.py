import os
import h5py
import numpy as np
from pyTNG import data_interface as _data_interface
from pyTNG.cosmology import TNGcosmo
from astropy import units as u


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


def get_halo(df, snap, halo_id):
    if "snap" in df.keys():
        halo = df[(df.Halo_id == halo_id) & (df.snap == snap)]
    else:
        halo = df[df.Halo_id == halo_id]
    return halo


def get_haloID_from_galaxyID(df, galaxy_id, snap):
    halo = df[(df.idx == galaxy_id) & (df.snap == snap)]
    halo_id = halo.Halo_id.values[0]
    return halo_id


def get_galaxyID_from_haloID(df, halo_id, snap):
    halo = df[(df.Halo_id == halo_id) & (df.snap == snap)]
    galaxy_id = halo.idx.values[0]
    return galaxy_id


def get_halo_data(df, halo_id, snap):
    data = {}
    halo = get_halo(df, snap, halo_id)
    data["snapshot"] = int(halo.snap)
    data["redshift"] = get_redshift(data["snapshot"])
    data["r_vir"] = halo.R_vir.values[0] / TNGcosmo.h / (1 + data["redshift"])
    data["time"] = TNGcosmo.age(data["redshift"])
    data["Galaxy_Mstar"] = halo.Galaxy_M_star.values[0] * 1e10 / TNGcosmo.h
    data["Galaxy_Mgas"] = halo.Galaxy_M_gas.values[0] * 1e10 / TNGcosmo.h
    data["Fraction_Mstar_in_Galaxy"] = halo.Galaxy_star_fraction.values[0]
    data["Galaxy_SFR"] = halo.Galaxy_SFR.values[0]
    data["Galaxy_idx"] = halo.idx.values[0]
    data["Halo_idx"] = halo_id
    return data


def autozoom(r_vir, gal_hmr, factor=20):
    zoom_in = int(np.ceil(r_vir / gal_hmr / factor))
    if zoom_in > 1:
        return zoom_in
    else:
        return 1
