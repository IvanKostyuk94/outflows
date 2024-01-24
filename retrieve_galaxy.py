import os
import numpy as np
import pandas as pd
import pyTNG.utils as utils
from pyTNG import data_interface as _data_interface
from pyTNG.cosmology import TNGcosmo

h = TNGcosmo.h


def get_sim():
    basepath = "/virgotng/universe/IllustrisTNG/"
    sim_name = "L35n2160TNG"
    sim = _data_interface.TNG50Simulation(os.path.join(basepath, sim_name))
    sim_path = os.path.join(basepath, sim_name, "output")
    return sim, sim_path


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


def get_subhalo_df(sim, snap_num):
    dataset = next(sim.group_cat[snap_num].chunk_generator("subhalo"))
    keys_needed = [
        "SubhaloPos",
        "SubhaloVel" "SubhaloWindMass" "SubhaloSFR" "SubhaloMassType",
    ]
    sub_dict = {key: dataset[key] for key in keys_needed}
    dataset_df = utils.dfFromArrDict(sub_dict)
    return dataset_df


def reduce_df(df):
    filt = df[("GroupMassType", 4)] * 1e10 / h > 7
    reduced_df = df[filt]

    new_df = pd.DataFrame().assign(
        Halo_pos_x=reduced_df[("SubhaloPos", 0)],
        Halo_pos_y=reduced_df[("SubhaloPos", 1)],
        Halo_pos_z=reduced_df[("SubhaloPos", 2)],
        Halo_M=reduced_df[("GroupMass", 0)],
        Halo_M_gas=reduced_df[("GroupMassType", 0)],
        Halo_M_star=reduced_df[("GroupMassType", 4)],
        R_vir=reduced_df[("Group_R_Crit200", 0)],
    )
    return new_df
