import os
import numpy as np
import pandas as pd
import pyTNG.utils as utils
from pyTNG.cosmology import TNGcosmo
from config import config
from utils import get_sim

h = TNGcosmo.h


def get_halo_df(sim, snap_num):
    dataset = next(sim.group_cat[snap_num].chunk_generator("halo"))
    keys_needed = [
        "GroupFirstSub",
        "Group_R_Crit200",
        "GroupPos",
        "GroupMass",
        "GroupMassType",
    ]
    sub_dict = {key: dataset[key] for key in keys_needed}
    dataset_df = utils.dfFromArrDict(sub_dict)
    return dataset_df


def get_galaxy_df(sim, snap_num):
    dataset = next(sim.group_cat[snap_num].chunk_generator("subhalo"))
    keys_needed = [
        "SubhaloPos",
        "SubhaloVel",
        "SubhaloWindMass",
        "SubhaloSFR",
        "SubhaloMassType",
        "SubhaloHalfmassRadType",
        "SubhaloBHMdot",
    ]
    sub_dict = {key: dataset[key] for key in keys_needed}
    dataset_df = utils.dfFromArrDict(sub_dict)
    return dataset_df


def reduce_halo_df(df):
    filt = np.log10(df[("GroupMassType", 4)] * 1e10 / h) > 7
    reduced_df = df[filt]

    new_df = pd.DataFrame().assign(
        Halo_pos_x=reduced_df[("GroupPos", 0)],
        Halo_pos_y=reduced_df[("GroupPos", 1)],
        Halo_pos_z=reduced_df[("GroupPos", 2)],
        Halo_M=reduced_df[("GroupMass", 0)],
        Halo_M_gas=reduced_df[("GroupMassType", 0)],
        Halo_M_star=reduced_df[("GroupMassType", 4)],
        R_vir=reduced_df[("Group_R_Crit200", 0)],
        Galaxy_id=reduced_df[("GroupFirstSub", 0)],
    )
    return new_df


def reduce_galaxy_df(df):
    filt = np.log10(df[("SubhaloMassType", 4)] * 1e10 / h) > 7
    reduced_df = df[filt]

    new_df = pd.DataFrame().assign(
        Galaxy_pos_x=reduced_df[("SubhaloPos", 0)],
        Galaxy_pos_y=reduced_df[("SubhaloPos", 1)],
        Galaxy_pos_z=reduced_df[("SubhaloPos", 2)],
        Galaxy_M_gas=reduced_df[("SubhaloMassType", 0)],
        Galaxy_M_star=reduced_df[("SubhaloMassType", 4)],
        Galaxy_M_wind=reduced_df[("SubhaloWindMass", 0)],
        Galaxy_SFR=reduced_df[("SubhaloSFR", 0)],
        Galaxy_HMR=reduced_df[("SubhaloHalfmassRadType", 4)],
        Galaxy_vel_x=reduced_df[("SubhaloVel", 0)],
        Galaxy_vel_y=reduced_df[("SubhaloVel", 1)],
        Galaxy_vel_z=reduced_df[("SubhaloVel", 2)],
        BH_growth=reduced_df[("SubhaloBHMdot", 0)],
    )
    return new_df


def get_reduced_df(snap, type="halo"):
    sim, _ = get_sim()
    if type == "halo":
        df = get_halo_df(sim, snap)
        df = reduce_halo_df(df)
    elif type == "galaxy":
        df = get_galaxy_df(sim, snap)
        df = reduce_galaxy_df(df)
    else:
        raise NotImplementedError(f"{type} is not implemented")
    return df


def match_with_galaxy(halo_df, galaxy_df):
    halo_df["Halo_id"] = halo_df.index
    halo_df.set_index("Galaxy_id", inplace=True)
    full_df = halo_df.join(galaxy_df, how="inner")
    full_df["Galaxy_star_fraction"] = (
        full_df["Galaxy_M_star"] / full_df["Halo_M_star"]
    )
    full_df["Galaxy_gas_fraction"] = (
        full_df["Galaxy_M_gas"] / full_df["Halo_M_gas"]
    )
    return full_df


def build_full_df(snap):
    halo_df = get_reduced_df(snap, type="halo")
    galaxy_df = get_reduced_df(snap, type="galaxy")
    full_df = match_with_galaxy(halo_df, galaxy_df)
    return full_df


def generate_database(snap):
    base_path = config["base_path"]
    dir = config["dir_prefix"] + str(snap)
    df_name = config["df_name"] + str(snap) + config["hdf_ending"]
    dir_path = os.path.join(base_path, dir)
    if not os.path.exists(dir_path):
        os.system(f"mkdir {dir_path}")
    save_path = os.path.join(dir_path, df_name)
    full_df = build_full_df(snap)
    full_df.to_hdf(save_path, "galaxies")
    return
