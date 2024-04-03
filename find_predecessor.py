import h5py
import os
import numpy as np
import pandas as pd
from config import config
from utils import dist_to_cm


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


def get_history_tuple_prop(idx, snap, history, prop):
    ID_raw = int(snap * 1e12 + idx)
    id_array = np.array(history["SubhaloIDRaw"])
    subhalo_pos = int(np.where(id_array == ID_raw)[0])
    return history[prop[0]][subhalo_pos, prop[1]]


def create_z_snap_dict(df):
    z_snap = {}
    for i, z in enumerate(df.z.unique()):
        z_snap[z] = i
    return z_snap


def add_snap_column(df):
    z_snap = create_z_snap_dict(df)
    df["snap"] = [z_snap[z] for z in df.z]
    return


def add_progenitors(
    df_name=config["full_df_name"],
    base_path=config["base_path"],
    merger_history_path=config["merger_history_path"],
    file_ending=config["df_ending"],
):
    df_path = os.path.join(base_path, df_name + file_ending)
    df = pd.read_pickle(df_path)

    add_snap_column(df)
    history = h5py.File(merger_history_path)

    id_array = np.array(history["SubhaloIDRaw"])
    progenitor_idxs = np.array(history["SubhaloID"])

    df["progenitor"] = None
    df["progenitor_snap"] = None
    counter = 0

    for i, element in df.iterrows():
        snap = element.snap
        idx = element.idx
        progenitor_snap, progenitor_id = get_history_prop(
            idx,
            snap,
            history,
            id_array=id_array,
            progenitor_array=progenitor_idxs,
            prop="FirstProgenitorID",
        )
        df.loc[i, "progenitor"] = progenitor_id
        df.loc[i, "progenitor_snap"] = progenitor_snap

        counter += 1
        if (counter % 1000) == 0:
            print(f"{counter/len(df)*100}% done")
    df.to_pickle(df_path)
    return


# Just a testing function to make sure the correct galaxy is selected
def testing_function(
    df_name=config["df_name"],
    base_path=config["base_path"],
    merger_history_path=config["merger_history_path"],
    file_ending=config["hdf_ending"],
):
    df_path = os.path.join(base_path, df_name + file_ending)
    df = pd.read_pickle(df_path)
    df = df[100000:]

    add_snap_column(df)
    history = h5py.File(merger_history_path)

    sample = df.sample(10)

    for _, element in sample.iterrows():
        snap = element.snap
        idx = element.idx
        r = get_history_tuple_prop(
            idx, snap, history, prop=["SubhaloHalfmassRadType", 4]
        )
        pos = get_history_prop(idx, snap, history, prop="GroupPos")

        print(element.r / dist_to_cm(element.z))
        print(element.Halo_pos_x)
        print(element.Halo_pos_y)
        print(element.Halo_pos_z)
        print("-" * 80)
    return