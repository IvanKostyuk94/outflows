import numpy as np
from Grid_halo import grid_gas, get_halo
from plotting import plot_prop_maps
from config import config
from find_progenitors import get_progenitor_history


def get_halo_from_galaxy_id(df, galaxy_id, snap):
    halo = df[(df.idx == galaxy_id) & (df.snap == snap)]
    return halo


def retrieve_history_maps(
    halo_id,
    full_df,
    snap,
    prop,
    grid_size=100,
    zoom_in=1,
    out_only=False,
    angle=None,
    v_out_threshold=None,
    v_esc_ratio=None,
):

    halo = get_halo(df=full_df, snap=snap, halo_id=halo_id)

    galaxy_id = int(halo.idx)
    galaxy_idces, snap_nums = get_progenitor_history(
        galaxy_idx=galaxy_id, snap_num=snap
    )

    for galaxy_id, snap in zip(galaxy_idces, snap_nums):
        halo = get_halo_from_galaxy_id(full_df, galaxy_id, snap)
        try:
            halo_id = int(halo.Halo_id)
        except TypeError:
            break
        if zoom_in == "autozoom":
            zoom_in = int(np.ceil(halo.R_vir / halo.Galaxy_HMR / 20))

        gas = grid_gas(
            halo_id,
            full_df,
            snap,
            out_only=out_only,
            threshold_velocity=v_out_threshold,
            v_esc_ratio=v_esc_ratio,
            grid_size=grid_size,
            zoom_in=zoom_in,
            projection_angle=angle,
        )

        r_vir = float(halo.R_vir)
        if zoom_in != 1:
            with_circle = False
            box_size = r_vir * 2 * float(config["cutout_scale"]) / zoom_in
            sizebar_length = 1
        else:
            with_circle = True
            box_size = r_vir * 2 / zoom_in
            sizebar_length = 10
        plot_prop_maps(
            gas, r_vir, prop, snap, with_circle, box_size, sizebar_length
        )
    return
