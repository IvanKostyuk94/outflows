import numpy as np
import pandas as pd
import os
import illustris_python as il
from utils import dfFromArrDict, get_sim_path
from config import config


def get_metalliciy_df(sim_path, snap_num):
    dataset = il.groupcat.loadSubhalos(sim_path, snap_num)
    keys_needed = [
        "SubhaloGasMetallicity",
        "SubhaloGasMetallicityHalfRad",
        "SubhaloGasMetallicitySfr",
    ]
    sub_dict = {key: dataset[key] for key in keys_needed}
    dataset_df = dfFromArrDict(sub_dict)
    return dataset_df


def get_v_df(sim_path, snap_num):
    dataset = il.groupcat.loadSubhalos(sim_path, snap_num)
    keys_needed = [
        "SubhaloVelDisp",
        "SubhaloVmax",
    ]
    sub_dict = {key: dataset[key] for key in keys_needed}
    dataset_df = dfFromArrDict(sub_dict)
    return dataset_df


def add_quantities(df_name, type="Metallicity"):
    sim_path = get_sim_path()
    df_path = os.path.join(config["base_path"], df_name)
    df = pd.read_hdf(df_path)
    for snap in df.snap.unique():
        print(f"Working on snap {snap}")
        sub_df = df[df["snap"] == snap]
        if type == "Metallicity":
            data_df = get_metalliciy_df(sim_path, snap_num=snap)
        elif type == "Velocities":
            data_df = get_v_df(sim_path, snap_num=snap)
        else:
            raise NotImplementedError(f"{type} is not implemented yet")
        print("Finished creating df")
        if snap == 2:
            for key in data_df:
                if key not in df.keys():
                    df[key[0]] = np.nan * np.ones(len(df))
        for _, gal in sub_df.iterrows():
            for key in data_df.keys():
                df.loc[(df.idx == gal.idx) & (df["snap"] == snap), key[0]] = (
                    data_df.loc[gal.idx][key]
                )
        df.to_hdf(df_path, config["hdf_key"])


if __name__ == "__main__":
    add_quantities("all_galaxies_extended.hdf5", type="Velocities")
