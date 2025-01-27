import os
import h5py
import pandas as pd
import numpy as np
from pyTNG import data_interface as _data_interface
from pyTNG.cosmology import TNGcosmo
from astropy import units as u
from astropy.constants import G
from config import config
import pickle
from astropy import constants as c
from sklearn.decomposition import PCA


def get_sim():
    basepath = config["tng_datapath"]  # "/virgotng/universe/IllustrisTNG/"
    sim_name = config["sim_name"]
    sim = _data_interface.TNG50Simulation(os.path.join(basepath, sim_name))
    sim_path = os.path.join(basepath, sim_name, "output")
    return sim, sim_path


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


def autozoom(r_vir, gal_ghmr, factor=10):
    zoom_in = int(np.ceil(r_vir / gal_ghmr / factor))
    if zoom_in > 1:
        return zoom_in
    else:
        return 1


# Corrects the particle dictionary to only contain the particles in relevant
def map_to_new_dict(particles, relevant):
    rel_particles = {}
    newcount_particles = (relevant).sum()
    for key, value in particles.items():
        if key == "count":
            continue
        try:
            rel_particles[key] = value[relevant]
        # for Python scalars
        except TypeError as e:
            if "not subscriptable" in str(e):
                pass
            else:
                raise
        # for numpy scalars
        except IndexError as e:
            if "invalid index to scalar variable" in str(e):
                pass
            else:
                print(key)
                raise
    if "count" in particles:
        rel_particles["count"] = newcount_particles
    return rel_particles


def sort_all_keys(particles, sort_key):
    sorted_idces = np.argsort(particles[sort_key])
    for key in particles.keys():
        if key == "count":
            continue
        sorted_array = particles[key][sorted_idces]
        particles[key] = sorted_array
    return


def calculate_acc(mass, dist_km):
    G_correct = G.to(u.km**3 / u.kg / u.s**2).value
    g = -1 * G_correct * get_mass_in_kg(mass) / dist_km**2
    return g

def get_mu(galaxy):
    H = galaxy['H']+galaxy['H+']
    H2 = galaxy['H2']+galaxy['H2+']
    He = galaxy['HE']+galaxy['HE+']+galaxy['HE++']
    mu = H+2*H2+4*He
    return mu 

def get_serra_sfr(galaxy):
    t_ff = np.sqrt(3*np.pi/(32*G*galaxy['rho']*u.Msun/u.kpc**3))
    mu = get_mu(galaxy)
    SFR = (0.1*galaxy['mass']*u.M_sun*galaxy['H2']*mu/t_ff).to('Msun/yr')
    return SFR.value

def get_serra_galaxy(snap, galaxy_id):
    base_path = config['base_path']
    snapdir = 'snap'+str(snap)
    galaxyname = str(galaxy_id) + '.pickle'
    galaxy_path = os.path.join(base_path, 'data_new', snapdir, galaxyname)
    with open(galaxy_path, 'rb') as f:
        galaxy = pickle.load(f)
    return galaxy

def get_serra_galaxy_stars(snap, galaxy_id):
    base_path = config['base_path']
    snapdir = 'snap'+str(snap)
    galaxyname = str(galaxy_id) + '.pickle'
    galaxy_path = os.path.join(base_path, 'star_data', snapdir, galaxyname)
    with open(galaxy_path, 'rb') as f:
        galaxy = pickle.load(f)
    return galaxy


def get_serra_gas(snap, galaxy_id):
    galaxy = get_serra_galaxy(snap, galaxy_id)
    mu = get_mu(galaxy)
    # galaxy['Density'] = galaxy['rho']* c.M_sun / c.m_p  / mu / (u.kpc.to(u.cm)) ** 3
    galaxy['Density'] = galaxy['rho']
    galaxy['Coordinates'] = galaxy['pos']
    galaxy['Temperture'] = galaxy['temp']
    galaxy['count'] = len(galaxy['mass'])
    galaxy['Velocities'] = galaxy['vel']
    galaxy['hsml']  = 2*galaxy['smooth']#*(3/4/np.pi)**(1/3)
    galaxy['Masses'] = galaxy['mass']
    galaxy['StarFormationRate'] = get_serra_sfr(galaxy)
    galaxy["GFM_Metallicity"] = galaxy["metal"]
    del galaxy['star_mass']
    del galaxy['z']
    return galaxy

def get_serra_stars(snap, galaxy_id):
    galaxy = get_serra_galaxy_stars(snap, galaxy_id)
    galaxy['Coordinates'] = galaxy['pos']
    galaxy['count'] = len(galaxy['mass'])
    galaxy['Velocities'] = galaxy['vel']
    galaxy['Masses'] = galaxy['mass']
    del galaxy['star_mass']
    del galaxy['z']
    return galaxy

def weighted_avg_and_std(values, weights, average):
    """
    Return the weighted average and standard deviation.

    They weights are in effect first normalized so that they 
    sum to 1 (and so they must not all be 0).

    values, weights -- NumPy ndarrays with the same shape.
    """
    velocities = np.linalg.norm(values, axis=1)
    average_vel = np.linalg.norm(average)

    variance = np.average((velocities-average_vel)**2, weights=weights)
    return np.sqrt(variance)

def get_galaxy_pos_vel(particles):
    keys_to_include = {'pos', 'vel', 'mass'}

# Dictionary comprehension to create the subset
    rel_particles = {key: particles[key] for key in keys_to_include if key in particles}
    rel_particles['rel_pos'] = np.linalg.norm(rel_particles['pos'], axis=1)
    
    rel_idx = rel_particles['rel_pos']<2.5
    rel_particles = map_to_new_dict(rel_particles, rel_idx)
    pos = np.average(rel_particles['pos'], axis=0, weights=rel_particles['mass'])
    vel = np.average(rel_particles['vel'], axis=0, weights=rel_particles['mass'])
    disp  = weighted_avg_and_std(rel_particles['vel'], rel_particles['mass'], vel)
    return pos, vel, disp

def build_serra_df(name='serra_base'):
    serra_path = "/ptmp/mpa/ivkos/outflows/data_new"
    serra_files = os.listdir(serra_path)
    df_path = os.path.join(config['base_path'], name + config['hdf_ending'])
    data_dict = {}  
    data_dict['snap'] = []
    data_dict['z'] = []
    data_dict['Galaxy_M_star'] = []
    data_dict['idx'] = []
    data_dict['Galaxy_pos_x'] = []
    data_dict['Galaxy_pos_y'] = []
    data_dict['Galaxy_pos_z'] = []
    data_dict['Galaxy_vel_x'] = []
    data_dict['Galaxy_vel_y'] = []
    data_dict['Galaxy_vel_z'] = []
    data_dict['SubhaloVelDisp'] = []
    
    for snapdir in serra_files:
        snap = int(snapdir[-2:])
        print(snap)
        galaxies = os.listdir(os.path.join(serra_path, snapdir))
        for galaxy_id in galaxies:
            file_path = os.path.join(serra_path, snapdir, galaxy_id)
            with open(file_path, 'rb') as f:
                galaxy = pickle.load(f)
            pos, vel, disp = get_galaxy_pos_vel(galaxy)
            data_dict['snap'].append(snap)
            # todo:
            data_dict['z'].append(galaxy['z'])
            try:
                data_dict['Galaxy_M_star'].append(galaxy['star_mass'])
            except:
                print(snap)
                print(galaxy_id)
                continue
            # data_dict['Galaxy_M_gas'].append(galaxy['M_gas'])
            idx = int(galaxy_id.split('.')[0])
            data_dict['idx'].append(idx)
            data_dict['Galaxy_pos_x'].append(pos[0])
            data_dict['Galaxy_pos_y'].append(pos[1])
            data_dict['Galaxy_pos_z'].append(pos[2])
            data_dict['Galaxy_vel_x'].append(vel[0])
            data_dict['Galaxy_vel_y'].append(vel[1])
            data_dict['Galaxy_vel_z'].append(vel[2])
            data_dict['SubhaloVelDisp'].append(disp)
    for key in data_dict.keys():
        data_dict[key] = np.array(data_dict[key])
    df = pd.DataFrame(data_dict)
    df['M_star_log'] = np.log10(df['Galaxy_M_star'])
    df.to_hdf(df_path, key=config["hdf_key"])
    return

def create_particle_box(gas, df, idx, z, stars=None):
    if gas["count"] < 5:
        if stars is not None:
            return 0, 0
        else:
            return 0
    pca = PCA(3)
    pca.fit(gas["Coordinates"])
    gas["Coordinates"] = pca.transform(gas["Coordinates"])


if __name__ == "__main__":
    build_serra_df()
    