import os
import warnings
import pandas as pd
import numpy as np
from tng_cosmo import TNGcosmo
from astropy import units as u
from astropy.constants import G
from config import config
from sklearn.decomposition import PCA


def get_redshift(snap_num):
    redshifts = {
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
        10: 7.23627606616736,
        11: 7.005417045544533,
        12: 6.491597745667503,
        13: 6.0107573988449,
        14: 5.846613747881867,
        15: 5.5297658079491026,
        16: 5.227580973127337,
        17: 4.995933468164624,
        18: 4.664517702470927,
        19: 4.428033736605549,
        20: 4.176834914726472,
        21: 4.0079451114652676,
        22: 3.7087742646422353,
        23: 3.4908613692606485,
        24: 3.2830330579565246,
        25: 3.008131071630377,
    }
    # sim, _ = get_sim()
    # z = sim.snap_cat[snap_num].header["Redshift"]
    return redshifts[snap_num]


def scale_factor(z):
    return 1 / (z + 1)


def get_snap_name(snap_num):
    return f"snap_{snap_num:03d}"


def get_snap_dir(snap_num):
    return f"snapdir_{snap_num:03d}"


def get_mass_in_kg(mass):
    msun_to_kg = (1 * u.M_sun).to(u.kg)
    mass = mass * 1e10 / TNGcosmo.h * msun_to_kg
    return mass.value


def get_dist_in_km(dist, z):
    kpc_to_km = (1 * u.kpc).to(u.km)
    dist = dist / (z + 1) / TNGcosmo.h * kpc_to_km
    return dist.value


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


def autozoom(r_vir, gal_ghmr, factor=10):
    zoom_in = int(np.ceil(r_vir / gal_ghmr / factor))
    if zoom_in > 1:
        return zoom_in
    else:
        return 1


def map_to_new_dict(particles, relevant):
    """Filter a particle dict by a boolean mask."""
    rel_particles = {}
    newcount_particles = relevant.sum()
    for key, value in particles.items():
        if key == "count":
            continue
        try:
            rel_particles[key] = value[relevant]
        except TypeError as e:
            if "not subscriptable" in str(e):
                pass
            else:
                raise
        except IndexError as e:
            if "invalid index to scalar variable" in str(e):
                pass
            else:
                raise
    if "count" in particles:
        rel_particles["count"] = newcount_particles
    return rel_particles


def sort_all_keys(particles, sort_key):
    sorted_idces = np.argsort(particles[sort_key])
    for key in particles.keys():
        if key == "count":
            continue
        particles[key] = particles[key][sorted_idces]


def calculate_acc(mass, dist_km):
    G_correct = G.to(u.km**3 / u.kg / u.s**2).value
    g = -1 * G_correct * get_mass_in_kg(mass) / dist_km**2
    return g


def create_particle_box(gas, df, idx, z, stars=None):
    if gas["count"] < 5:
        if stars is not None:
            return 0, 0
        else:
            return 0
    pca = PCA(3)
    pca.fit(gas["Coordinates"])
    gas["Coordinates"] = pca.transform(gas["Coordinates"])


def dfFromArrDict(arrDict):
    """Create a MultiIndex DataFrame from a dict of arrays.

    1D arrays become single-column entries; 2D arrays expand to one
    column per second-dimension index.
    """
    tuples = []
    keys = []
    problem_keys = []
    for key, arr in arrDict.items():
        try:
            sh = arr.shape
        except AttributeError:
            problem_keys.append(key)
            continue
        keys.append(key)
        if len(sh) == 1:
            tuples.append((key, 0))
        elif len(sh) == 2:
            tuples.extend([(key, i) for i in range(sh[1])])
        else:
            raise RuntimeError(
                "Got array with dimension > two at key: " + str(key)
            )

    if problem_keys:
        warnings.warn(
            "Could not integrate the following keys into dataframe: "
            + str(problem_keys)
            + "!\nMaybe the associated values are scalar"
            + " and this is actually expected?"
        )

    index = pd.MultiIndex.from_tuples(tuples)
    df = pd.DataFrame(columns=index)
    for key in keys:
        if arrDict[key].ndim == 1:
            df[key] = arrDict[key]
        else:
            df.loc[:, key] = arrDict[key]
    return df


