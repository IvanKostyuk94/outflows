from utils import (
    get_sim,
    get_redshift,
    scale_factor,
    get_dm_mass,
    get_mass_in_kg,
    get_dist_in_km,
)
import illustris_python as il
from pyTNG import gridding, gas_temperature
import numpy as np
from config import config
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R
from astropy.constants import G
from astropy import units as u
from scipy.integrate import cumtrapz
from gaussian_outflow_selection import group_gas


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


def get_galaxy_pos(df, halo_id):
    gal_pos = np.array(
        [
            float(df[df.Halo_id == halo_id].Galaxy_pos_x),
            float(df[df.Halo_id == halo_id].Galaxy_pos_y),
            float(df[df.Halo_id == halo_id].Galaxy_pos_z),
        ]
    )
    return gal_pos


def get_relative_coordinates(gas, gal_pos):
    gas["Relative_Coordinates"] = gas["Coordinates"] - gal_pos
    return


def get_relative_distances(gas):
    gas["Relative_Distances"] = np.sqrt(
        np.sum(np.square(gas["Relative_Coordinates"]), axis=1)
    )
    return


# Retrieves all particles in a halo and immidiatly adds the outflow velocity
def retrieve_halo_gas(df, snap, halo_id):
    # _, sim_path = get_sim()
    # gas = il.snapshot.loadHalo(sim_path, snap, halo_id, "gas")
    gas = get_gas_v_esc(df, snap, halo_id)
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
    gas["Relative_Velocities_abs"] = np.linalg.norm(
        gas["Relative_Velocities"], axis=1
    )
    galaxy_pos = get_galaxy_pos(df, halo_id)
    get_relative_coordinates(gas, galaxy_pos)
    gas["Direction"] = (
        gas["Relative_Coordinates"].T
        / np.linalg.norm(gas["Relative_Coordinates"], axis=1)
    ).T
    # >0 means outflow, <0 means infall
    gas["Flow_Velocities"] = np.float32(
        np.multiply(gas["Relative_Velocities"], gas["Direction"]).sum(axis=1)
    )
    gas["Rot_Velocities"] = np.sqrt(
        gas["Relative_Velocities_abs"] ** 2 - gas["Flow_Velocities"] ** 2
    )
    gas_temperature.gasTemp(gas)
    gas["hsml"] = 2.5 * (
        3 * (gas["Masses"] / gas["Density"]) / (4.0 * np.pi)
    ) ** (1.0 / 3)
    return gas


def calculate_acc(mass, dist_km):
    G_correct = G.to(u.km**3 / u.kg / u.s**2).value
    g = -1 * G_correct * get_mass_in_kg(mass) / dist_km**2
    return g


def sort_all_keys(particles, sort_key):
    sorted_idces = np.argsort(particles[sort_key])
    for key in particles.keys():
        sorted_array = particles[key][sorted_idces]
        particles[key] = sorted_array
    return


def get_gas_v_esc(df, snap, halo_id):
    _, sim_path = get_sim()
    gas = il.snapshot.loadHalo(sim_path, snap, halo_id, "gas")
    dm = il.snapshot.loadHalo(sim_path, snap, halo_id, "dm")
    stars = il.snapshot.loadHalo(sim_path, snap, halo_id, "stars")
    halo_particles = [gas, dm, stars]
    tot_num = np.sum(particles["count"] for particles in halo_particles)
    gal_center = get_galaxy_pos(df, halo_id)
    all_particles = {}
    all_particles["Masses"] = np.empty(tot_num)
    all_particles["Relative_Distances"] = np.empty(tot_num)
    all_particles["Numbering"] = np.empty(tot_num)

    for i, particles in enumerate(halo_particles):
        if i == 0:
            start = 0
        else:
            start = start + halo_particles[i - 1]["count"]
        end = particles["count"] + start
        get_relative_coordinates(particles, gal_center)
        get_relative_distances(particles)
        if "Masses" in particles.keys():
            all_particles["Masses"][start:end] = particles["Masses"]
        else:
            dm_mass = get_dm_mass(snap)
            all_particles["Masses"][start:end] = dm_mass * np.ones(
                particles["count"]
            )
        all_particles["Relative_Distances"][start:end] = particles[
            "Relative_Distances"
        ]
        particles["Numbering"] = np.arange(start, end)
        all_particles["Numbering"][start:end] = particles["Numbering"]

    sort_all_keys(particles=all_particles, sort_key="Relative_Distances")

    all_particles["Masses_Cum"] = np.cumsum(all_particles["Masses"])
    z = get_redshift(snap)
    all_particles["Relative_Distances_km"] = get_dist_in_km(
        all_particles["Relative_Distances"], z
    )
    all_particles["Grav_acc"] = calculate_acc(
        all_particles["Masses_Cum"], all_particles["Relative_Distances_km"]
    )
    all_particles["Grav_pot"] = cumtrapz(
        all_particles["Grav_acc"],
        x=all_particles["Relative_Distances_km"],
        initial=0,
    )
    all_particles["v_esc"] = np.sqrt(
        2
        * (
            all_particles["Grav_pot"]
            - all_particles["Grav_acc"][-1]
            * all_particles["Relative_Distances_km"][-1]
            - all_particles["Grav_pot"][-1]
        )
    )

    sort_all_keys(particles=all_particles, sort_key="Numbering")

    gas_indices = np.where(
        np.isin(all_particles["Numbering"], gas["Numbering"])
    )[0]

    gas["v_esc"] = all_particles["v_esc"][gas_indices]
    return gas


# Threshold velocities should be provided as absolute velocities in km/s
def select_outflowing_gas(gas, threshold_velocity=None, v_esc_ratio=None):
    if gas["count"] == 0:
        return gas
    else:
        if threshold_velocity is not None:
            idces_rel_gas = gas["Flow_Velocities"] > threshold_velocity
        else:
            idces_rel_gas = (
                gas["Relative_Velocities_abs"] > v_esc_ratio * gas["v_esc"]
            ) & (gas["Flow_Velocities"] > 0)
        rel_gas = map_to_new_dict(gas, idces_rel_gas)
    return rel_gas


def select_gas_group(gas, group_num):
    if gas["count"] == 0:
        return gas
    else:
        idces_rel_gas = gas["group"] == group_num
        rel_gas = map_to_new_dict(gas, idces_rel_gas)
    return rel_gas


def gal_plane_tranformer(particles, r_HMR):
    particles["Relative_Distances"] = np.sqrt(
        np.sum(np.square(particles["Relative_Coordinates"]), axis=1)
    )
    idces_rel_particles = particles["Relative_Distances"] < 2 * r_HMR
    rel_particles = map_to_new_dict(particles, idces_rel_particles)
    pca = PCA(3)
    pca.fit(rel_particles["Coordinates"])
    return pca


def rotate_into_galactic_plane(gas, center, r_HMR):
    gas["Original_Coordinates"] = gas["Coordinates"]
    transformer = gal_plane_tranformer(gas, r_HMR)
    gas["Coordinates"] = transformer.transform(gas["Coordinates"])
    gas["Velocities"] = transformer.transform(gas["Velocities"])

    center = transformer.transform(center)
    return center[0]


def line_of_sight_projection(gas, angle, center):
    los_dir = np.array([np.cos(angle), 0, np.sin(angle)])
    los_rot_matrix = [
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)],
    ]
    proj_velocities = np.float32(
        np.multiply(gas["Velocities"], los_dir).sum(axis=1)
    )
    gas["Projected_Velocities"] = proj_velocities
    rot_transformer = R.from_matrix(los_rot_matrix)
    gas["Coordinates"] = rot_transformer.apply(gas["Coordinates"])
    center = rot_transformer.apply(center)
    return center


def get_gridded(gas, quants, grid_shape, grid_size, grid_cen, n_threads):
    grids = gridding.depositParticlesOnGrid(
        gas_parts=gas,
        method="sphKernelDep",
        quants=quants,
        box_size_parts=[0, 0, 0],
        grid_shape=grid_shape,
        grid_size=grid_size,
        grid_cen=grid_cen,
        n_threads=n_threads,
    )

    grid_sfr = gridding.depositParticlesOnGrid(
        gas_parts=gas,
        method="sphKernelDep",
        quants=[],
        box_size_parts=[0, 0, 0],
        grid_shape=grid_shape,
        grid_size=grid_size,
        grid_cen=grid_cen,
        n_threads=n_threads,
        mass_key="StarFormationRate",
    )
    grids["StarFormationRate"] = grid_sfr["StarFormationRate"]
    return grids


def cut_zoomed(gas, r_vir, zoom_in):
    max_r = r_vir / zoom_in * np.sqrt(3)
    relevant_gas = np.linalg.norm(gas["Relative_Coordinates"], axis=1) < max_r
    selected_gas = map_to_new_dict(gas, relevant_gas)
    return selected_gas


def rot_preselection(gas, crit_ratio=0.5):
    relevant_gas = gas["Flow_Velocities"] / gas["Rot_Velocities"] > crit_ratio
    selected_gas = map_to_new_dict(gas, relevant_gas)
    return selected_gas


def grid_gas(
    halo_id,
    df,
    snap,
    out_only,
    threshold_velocity=None,
    v_esc_ratio=None,
    grid_size=100,
    n_threads=8,
    zoom_in=1,
    projection_angle=None,
    grouped_selection=False,
    group_props=["Flow_Velocities"],
    n_peak=None,
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
    r_HMR = float(df[df.Halo_id == halo_id].Galaxy_HMR)

    gas = retrieve_halo_gas(df, snap, halo_id)
    gas = cut_zoomed(gas=gas, r_vir=r_vir, zoom_in=zoom_in)
    if grouped_selection:
        gas = select_outflowing_gas(
            gas, threshold_velocity=0, v_esc_ratio=None
        )
        group_gas(gas, props=group_props, peak_number=n_peak)
    gal_center = rotate_into_galactic_plane(gas, [gal_center], r_HMR)
    if projection_angle is not None:
        gal_center = line_of_sight_projection(
            gas, projection_angle, gal_center
        )

    if out_only:
        gas = rot_preselection(gas)
        if threshold_velocity is None:
            if v_esc_ratio is not None:
                gas = select_outflowing_gas(
                    gas, threshold_velocity=None, v_esc_ratio=v_esc_ratio
                )
            else:
                raise ValueError(
                    'Either "v_esc_ratio" or "threshold_velocity" have to be not "None"'
                )
        else:
            if v_esc_ratio is None:
                gas = select_outflowing_gas(
                    gas,
                    threshold_velocity=threshold_velocity,
                    v_esc_ratio=None,
                )
            else:
                raise ValueError(
                    'Either "v_esc_ratio" or "threshold_velocity" have to be "None"'
                )

        # gas = select_outflowing_gas(gas, threshold_velocity)

    quants = [
        "Temperature",
        "GFM_Metallicity",
        "Flow_Velocities",
    ]
    if projection_angle is not None:
        quants.append("Projected_Velocities")
    if zoom_in == 1:
        box_size = (
            r_vir * 2 * float(config["cutout_scale"]) * np.ones(3) / zoom_in
        )
    else:
        box_size = r_vir * 2 * np.ones(3) / zoom_in
    shape = (grid_size * np.ones(3)).astype(np.int64)
    grid_cen = gal_center
    if grouped_selection is False:
        grids = get_gridded(
            gas=gas,
            quants=quants,
            grid_shape=shape,
            grid_size=box_size,
            grid_cen=grid_cen,
            n_threads=n_threads,
        )
        for quant in quants:
            grids[quant] = np.where(
                grids["Masses"] != 0, grids[quant] / grids["Masses"], 0
            )
        return grids
    else:
        all_grids = []
        for i in range(np.max(gas["group"]) + 1):
            gas_group = select_gas_group(gas, i)
            group_grids = get_gridded(
                gas=gas_group,
                quants=quants,
                grid_shape=shape,
                grid_size=box_size,
                grid_cen=grid_cen,
                n_threads=n_threads,
            )
            for quant in quants:
                group_grids[quant] = np.where(
                    group_grids["Masses"] != 0,
                    group_grids[quant] / group_grids["Masses"],
                    0,
                )
            all_grids.append(group_grids)
        return all_grids
