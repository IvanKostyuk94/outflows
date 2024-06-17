import numpy as np
import pandas as pd
import os
from pyTNG.utils import dfFromArrDict
from config import config
from utils import get_sim


def get_metalliciy_df(sim, snap_num):
    dataset = next(sim.group_cat[snap_num].chunk_generator("subhalo"))
    keys_needed = [
        "SubhaloGasMetallicity",
        "SubhaloGasMetallicityHalfRad",
        "SubhaloGasMetallicitySfr",
    ]
    sub_dict = {key: dataset[key] for key in keys_needed}
    dataset_df = dfFromArrDict(sub_dict)
    return dataset_df


def add_metallicities(df_name):
    sim, _ = get_sim()
    df_path = os.path.join(config["base_path"], df_name)
    df = pd.read_hdf(df_path)
    for snap in df.snap.unique():
        print(f"Working on snap {snap}")
        sub_df = df[df["snap"] == snap]
        metal_df = get_metalliciy_df(sim, snap_num=snap)
        print("Finished creating metal df")
        if snap == 2:
            for key in metal_df:
                if key not in df.keys():
                    df[key[0]] = np.nan * np.ones(len(df))
        for _, gal in sub_df.iterrows():
            for key in metal_df.keys():
                df.loc[(df.idx == gal.idx) & (df["snap"] == snap), key[0]] = (
                    metal_df.loc[gal.idx][key]
                )
        df.to_hdf(df_path, config["hdf_key"])
