from utils import (
    get_halo,
    map_to_new_dict,
)
import numpy as np
from config import config
from pyTNG import gridding
from process_gas import Galaxy

from scipy.spatial.transform import Rotation as R
from gaussian_outflow_selection import (
    group_gas,
    select_galaxy_group,
    get_only_outflowing_gas,
)


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


def remove_nans(grids, quants):
    for quant in quants:
        grids[quant] = np.where(
            grids["Masses"] != 0, grids[quant] / grids["Masses"], 0
        )
    return


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
    group_props=None,
    n_peak=4,
):
    gal = Galaxy(
        df=df,
        halo_id=halo_id,
        snap=snap,
        n_peak=n_peak,
        group_props=group_props,
    )

    gal.retrieve_halo_gas()
    gal.cut_gal_scale()
    # gal.rotate_into_galactic_plane()

    if out_only:
        gal.select_outflowing_gas(
            threshold_velocity=threshold_velocity,
            v_esc_ratio=v_esc_ratio,
        )

    quants = [
        "Temperature",
        "GFM_Metallicity",
        "Flow_Velocities",
    ]

    if zoom_in == 1:
        box_size = (
            gal.r_vir
            * 2
            * float(config["cutout_scale"])
            * np.ones(3)
            / zoom_in
        )
    else:
        box_size = gal.r_vir * 2 * np.ones(3) / zoom_in

    shape = (grid_size * np.ones(3)).astype(np.int64)
    grid_cen = np.array([0, 0, 0])
    if not grouped_selection:
        gal.preprocess_for_gridding()
        grids = get_gridded(
            gas=gal.gas,
            quants=quants,
            grid_shape=shape,
            grid_size=box_size,
            grid_cen=grid_cen,
            n_threads=n_threads,
        )
        remove_nans(grids, quants)
        return grids
    else:
        all_grids = []
        gal.select_outflowing_gas(threshold_velocity=0)
        gal.preprocess_for_gridding()

        full_gridded = get_gridded(
            gas=gal.gas,
            quants=quants,
            grid_shape=shape,
            grid_size=box_size,
            grid_cen=grid_cen,
            n_threads=n_threads,
        )
        all_grids.append(full_gridded)

        gal.group_gas()
        gal.get_gas_groups()
        gal.select_galaxy_group()
        galaxy_gridded = get_gridded(
            gas=gal.galaxy_group,
            quants=quants,
            grid_shape=shape,
            grid_size=box_size,
            grid_cen=grid_cen,
            n_threads=n_threads,
        )
        all_grids.append(galaxy_gridded)

        gal.get_only_outflowing_gas()
        outflow_gridded = get_gridded(
            gas=gal.grouped_out_gas,
            quants=quants,
            grid_shape=shape,
            grid_size=box_size,
            grid_cen=grid_cen,
            n_threads=n_threads,
        )
        all_grids.append(outflow_gridded)
        for grids in all_grids:
            remove_nans(grids, quants)
        return all_grids
