from utils import get_sim, get_redshift, scale_factor
import illustris_python as il
from pyTNG import gridding, gas_temperature
import pandas as pd
import numpy as np
from config import config
from sklearn.decomposition import PCA


# Corrects the particle dictionary to only contain the particles in relevant
def map_to_new_dict(particles, relevant):
    rel_particles = {}
    newcount_particles = (relevant).sum()
    for key, value in particles.items():
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


# Retrieves all particles in a halo and immidiatly adds the outflow velocity
def retrieve_halo_gas(df, snap, halo_id):
    _, sim_path = get_sim()
    gas = il.snapshot.loadHalo(sim_path, snap, halo_id, "gas")
    z = get_redshift(4)
    gas["Velocities"] = gas["Velocities"] * np.sqrt(scale_factor(z))
    galaxy_vel = np.array(
        [
            float(df[df.Halo_id == halo_id].Galaxy_vel_x),
            float(df[df.Halo_id == halo_id].Galaxy_vel_y),
            float(df[df.Halo_id == halo_id].Galaxy_vel_z),
        ]
    )
    gas["Relative_Velocities"] = gas["Velocities"] - galaxy_vel.T
    galaxy_pos = np.array(
        [
            float(df[df.Halo_id == halo_id].Galaxy_pos_x),
            float(df[df.Halo_id == halo_id].Galaxy_pos_y),
            float(df[df.Halo_id == halo_id].Galaxy_pos_z),
        ]
    )
    gas["Relative_Coordinates"] = gas["Coordinates"] - galaxy_pos.T
    gas["Direction"] = (
        gas["Relative_Coordinates"].T
        / np.linalg.norm(gas["Relative_Coordinates"], axis=1)
    ).T
    gas["Flow_Velocities"] = np.float32(
        np.multiply(gas["Relative_Velocities"], gas["Direction"]).sum(axis=1)
    )
    gas_temperature.gasTemp(gas)
    gas["hsml"] = 2.5 * (
        3 * (gas["Masses"] / gas["Density"]) / (4.0 * np.pi)
    ) ** (1.0 / 3)

    gas["Relative_Velocities_x"] = np.float32(gas["Relative_Velocities"][:, 0])
    gas["Relative_Velocities_y"] = np.float32(gas["Relative_Velocities"][:, 1])
    gas["Relative_Velocities_z"] = np.float32(gas["Relative_Velocities"][:, 2])
    return gas


# Threshold velocities should be provided as absolute velocities in km/s
def select_outflowing_gas(gas, threshold_velocity):
    if gas["count"] == 0:
        return gas
    else:
        idces_rel_gas = gas["Flow_Velocities"] > threshold_velocity
        rel_gas = map_to_new_dict(gas, idces_rel_gas)
    return rel_gas


def gal_plane_tranformer(particles, r_vir):
    particles["Relative_Distances"] = np.sqrt(
        np.sum(np.square(particles["Relative_Coordinates"]), axis=1)
    )
    idces_rel_particles = particles["Relative_Distances"] < 0.015 * r_vir
    rel_particles = map_to_new_dict(particles, idces_rel_particles)
    pca = PCA(3)
    pca.fit(rel_particles["Coordinates"])
    return pca


def rotate_into_galactic_plane(gas, center, r_vir):
    gas["Original_Coordinates"] = gas["Coordinates"]
    # pca = PCA(3)
    # pca.fit(gas["Coordinates"])
    transformer = gal_plane_tranformer(gas, r_vir)
    gas["Coordinates"] = transformer.transform(gas["Coordinates"])
    center = transformer.transform(center)
    return center[0]


def grid_gas(
    halo_id,
    df,
    snap,
    out_only,
    threshold_velocity=100,
    grid_size=100,
    n_threads=8,
):
    idx = halo_id
    halo_id = idx
    gal_center = np.array(
        [
            float(df[df.Halo_id == halo_id].Galaxy_pos_x),
            float(df[df.Halo_id == halo_id].Galaxy_pos_y),
            float(df[df.Halo_id == halo_id].Galaxy_pos_z),
        ]
    )
    r_vir = float(df[df.Halo_id == halo_id].R_vir)

    gas = retrieve_halo_gas(df, snap, halo_id)
    gal_center = rotate_into_galactic_plane(gas, [gal_center], r_vir)
    if out_only:
        gas = select_outflowing_gas(gas, threshold_velocity)

    quants = [
        "Temperature",
        "GFM_Metallicity",
        "Relative_Velocities_x",
        "Relative_Velocities_y",
        "Relative_Velocities_z",
        "Flow_Velocities",
    ]
    box_size = r_vir * 2 * float(config["cutout_scale"]) * np.ones(3) / 2
    shape = (grid_size * np.ones(3)).astype(np.int64)
    grid_cen = gal_center
    grids = gridding.depositParticlesOnGrid(
        gas_parts=gas,
        method="sphKernelDep",
        quants=quants,
        box_size_parts=[0, 0, 0],
        grid_shape=shape,
        grid_size=box_size,
        grid_cen=grid_cen,
        n_threads=n_threads,
    )

    grid_sfr = gridding.depositParticlesOnGrid(
        gas_parts=gas,
        method="sphKernelDep",
        quants=[],
        box_size_parts=[0, 0, 0],
        grid_shape=shape,
        grid_size=box_size,
        grid_cen=grid_cen,
        n_threads=n_threads,
        mass_key="StarFormationRate",
    )
    grids["StarFormationRate"] = grid_sfr["StarFormationRate"]

    for quant in quants:
        grids[quant] = np.where(
            grids["Masses"] != 0, grids[quant] / grids["Masses"], 0
        )
    return grids
