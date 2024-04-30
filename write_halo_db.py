import os
import pickle
import numpy as np
from utils import (
    get_halo_data,
    get_halo,
    get_haloID_from_galaxyID,
    get_galaxyID_from_haloID,
    autozoom,
)
from Grid_halo import (
    retrieve_halo_gas,
    cut_zoomed,
    select_gas_group,
    select_outflowing_gas,
)
from gaussian_outflow_selection import (
    group_gas,
    select_galaxy_group,
    get_only_outflowing_gas,
)
from find_progenitors import get_progenitor_history
from config import config


def select_keys_of_interest(gas, keys=None):
    if keys is None:
        keys = [
            "Coordinates",
            "Velocities",
            "Masses",
            "GFM_Metallicity",
            "Temperature",
            "StarFormationRate",
            "Flow_Velocities",
            "hsml",
        ]
    new_gas = {}
    for key, value in gas.items():
        if key in keys:
            new_gas[key] = value
    return new_gas


def create_halo_dict(
    df,
    halo_id,
    snap,
    zoom_in="autozoom",
    n_peak=4,
    group_props=None,
):
    data_dict = {}

    gas = retrieve_halo_gas(df=df, snap=snap, halo_id=halo_id)

    halo = get_halo(df=df, snap=snap, halo_id=halo_id)
    r_vir = float(halo.R_vir)

    if zoom_in == "autozoom":
        zoom_in = autozoom(halo.R_vir, halo.Galaxy_HMR)

    halo_info = get_halo_data(df, halo_id, snap)
    halo_info["zoom"] = zoom_in

    data_dict["info"] = halo_info

    gas = cut_zoomed(gas=gas, r_vir=r_vir, zoom_in=zoom_in)
    data_dict["full_galaxy"] = select_keys_of_interest(gas)

    out_gas = select_outflowing_gas(
        gas, threshold_velocity=0, v_esc_ratio=None
    )

    if group_props is None:
        group_props = [
            "Flow_Velocities",
            "Rot_Velocities",
            "Temperature",
            "Coordinates",
        ]

    group_gas(out_gas, props=group_props, peak_number=n_peak)

    gas_groups = []
    for i in range(np.max(out_gas["group"]) + 1):
        gas_group = select_gas_group(out_gas, i)
        gas_groups.append(gas_group)

    galaxy_group = select_galaxy_group(gas_groups)
    out_gas = get_only_outflowing_gas(out_gas, galaxy_group)
    data_dict["all_outflow_gas"] = select_keys_of_interest(out_gas)

    group_counter = 0
    for i, group in enumerate(gas_groups):
        if i != galaxy_group:
            group_name = f"outflow_gas_group{group_counter}"
            data_dict[group_name] = select_keys_of_interest(group)
            group_counter += 1
    return data_dict


def write_halo_db(
    df,
    halo_id,
    snap,
    zoom_in="autozoom",
    n_peak=4,
    group_props=None,
    with_history=False,
):

    if with_history:
        galaxy_idx = get_galaxyID_from_haloID(
            df=df, halo_id=halo_id, snap=snap
        )
        galaxy_idces, snap_nums = get_progenitor_history(
            galaxy_idx=galaxy_idx, snap_num=snap
        )
        full_data_dict = {}
        for idx, snap_num in zip(galaxy_idces, snap_nums):
            try:
                halo_idx = get_haloID_from_galaxyID(
                    df=df, galaxy_id=idx, snap=snap_num
                )
            except IndexError:
                break
            full_data_dict[snap_num] = create_halo_dict(
                df,
                halo_idx,
                snap_num,
                zoom_in=zoom_in,
                n_peak=n_peak,
                group_props=group_props,
            )

    else:
        full_data_dict = create_halo_dict(
            df,
            halo_id,
            snap,
            zoom_in=zoom_in,
            n_peak=n_peak,
            group_props=group_props,
        )
    if with_history:
        filename = f"{halo_id}_history.pickle"
    else:
        filename = f"{halo_id}.pickle"
    file_path = os.path.join(
        config["base_path"], config["dir_prefix"] + str(snap), filename
    )
    filehandler = open(file_path, "wb")
    pickle.dump(full_data_dict, filehandler)
    return
