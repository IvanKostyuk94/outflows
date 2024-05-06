import os
import numpy as np
import pandas as pd
from config import config
from Grid_halo import sort_all_keys, retrieve_halo_gas, get_relative_distances


def compute_sfr_radius(particles, tot_sfr):
    reduced_particles = {}
    relevant_keys = ["Relative_Distances", "StarFormationRate"]
    for key in relevant_keys:
        reduced_particles[key] = particles[key]
    sort_all_keys(particles=reduced_particles, sort_key="Relative_Distances")
    sfr_cum = np.cumsum(reduced_particles["StarFormationRate"])
    index_SFR = np.searchsorted(sfr_cum, tot_sfr / 2, side="right")
    try:
        r_SFR = reduced_particles["Relative_Distances"][index_SFR]
    except IndexError:
        r_SFR = np.nan
    return r_SFR


def get_sfr_radius(df, halo_id, snap):
    halo = df[(df.Halo_id == halo_id) & (df.snap == snap)]
    gal_sfr = float(halo.Galaxy_SFR)
    gas = retrieve_halo_gas(df, snap, halo_id)
    get_relative_distances(gas)
    r_SFR = compute_sfr_radius(particles=gas, tot_sfr=gal_sfr)
    return r_SFR


def add_sfr_radius_column(df, snap=None, save_path=None):
    if "r_SFR" not in df.keys():
        df["r_SFR"] = np.nan * np.ones(len(df))
    if snap is not None:
        iteration_df = df[df.snap == snap]
    else:
        iteration_df = df
    counter = 0
    for _, element in iteration_df.iterrows():
        halo_id = int(element.Halo_id)
        snap = int(element.snap)
        r_SFR = get_sfr_radius(df=iteration_df, halo_id=halo_id, snap=snap)
        if r_SFR == np.nan:
            print(f"No radius was found for halo {halo_id} in snap {snap}")
        df.loc[(df.snap == snap) & (df.Halo_id == halo_id), "r_SFR"] = r_SFR
        if counter % 100 == 0:
            print(f"Processed {counter/len(df)*100:.2f}% of galaxies")
        counter += 1
        if counter % 1000 == 0:
            if save_path is not None:
                df.to_hdf(save_path, "galaxies")
                df = pd.read_hdf(save_path)
    return


def update_df_with_rSFR(df_name):
    base_path = config["base_path"]
    path = os.path.join(base_path, df_name)
    df = pd.read_hdf(path)
    add_sfr_radius_column(df, save_path=path)
    df.to_hdf(path, "galaxies")
    return


if __name__ == "__main__":
    update_df_with_rSFR("all_galaxies_new.hdf5")
