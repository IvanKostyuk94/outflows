import h5py
import os
import numpy as np
import pandas as pd
from config import config


def get_history_prop(
    idx,
    snap,
    history,
    id_array=None,
    progenitor_array=None,
    prop="FirstProgenitorID",
):
    ID_raw = int(snap * 1e12 + idx)
    if id_array is None:
        id_array = np.array(history["SubhaloIDRaw"])
    subhalo_pos = int(np.where(id_array == ID_raw)[0])
    progenitor_id = history[prop][subhalo_pos]
    if progenitor_id == -1:
        snap = None
        galaxy_id = None
    else:
        progentior_idx = int(np.where(progenitor_array == progenitor_id)[0])
        progenitor_raw_id = history["SubhaloIDRaw"][progentior_idx]
        snap = int(np.round(progenitor_raw_id / 1e12))
        galaxy_id = int(progenitor_raw_id - 1e12 * snap)
    return snap, galaxy_id


def get_progenitor_history(
    galaxy_idx,
    snap_num,
    merger_history_path=config["merger_history_path"],
):
    history = h5py.File(merger_history_path)

    id_array = np.array(history["SubhaloIDRaw"])
    progenitor_idxs = np.array(history["SubhaloID"])

    galaxy_idces = [int(galaxy_idx)]
    snap_nums = [snap_num]

    while galaxy_idx is not None:
        progenitor_snap, progenitor_id = get_history_prop(
            galaxy_idx,
            snap_num,
            history,
            id_array=id_array,
            progenitor_array=progenitor_idxs,
            prop="FirstProgenitorID",
        )
        snap_num = progenitor_snap
        galaxy_idx = progenitor_id
        if galaxy_idx is not None:
            galaxy_idces.append(galaxy_idx)
            snap_nums.append(snap_num)
    return galaxy_idces, snap_nums


# Just a testing function to make sure the correct galaxy is selected
# def testing_function(
#     df_name=config["df_name"],
#     base_path=config["base_path"],
#     merger_history_path=config["merger_history_path"],
#     file_ending=config["hdf_ending"],
# ):
#     df_path = os.path.join(base_path, df_name + file_ending)
#     df = pd.read_pickle(df_path)
#     df = df[100000:]

#     add_snap_column(df)
#     history = h5py.File(merger_history_path)

#     sample = df.sample(10)

#     for _, element in sample.iterrows():
#         snap = element.snap
#         idx = element.idx
#         r = get_history_tuple_prop(
#             idx, snap, history, prop=["SubhaloHalfmassRadType", 4]
#         )
#         pos = get_history_prop(idx, snap, history, prop="GroupPos")

#         print(element.Halo_pos_x)
#         print(element.Halo_pos_y)
#         print(element.Halo_pos_z)
#         print("-" * 80)
#     return
