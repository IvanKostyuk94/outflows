import h5py
import os
import numpy as np
import pandas as pd
from config import config
from process_gas import Galaxy
from pyTNG.cosmology import TNGcosmo
from utils import get_redshift


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
        snap = np.nan
        galaxy_id = np.nan
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

    while ~np.isnan(galaxy_idx):
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
        if ~np.isnan(galaxy_idx):
            galaxy_idces.append(galaxy_idx)
            snap_nums.append(snap_num)
    return galaxy_idces, snap_nums


def get_progenitor_history_dict(
    df,
    galaxy_idx,
    snap_num,
    merger_history_path=config["merger_history_path"],
):
    galaxy_dict = {}
    idces, snaps = get_progenitor_history(galaxy_idx, snap_num, merger_history_path)
    for idx, snap in zip(idces, snaps):
        for key in df.keys():
            if key not in galaxy_dict.keys():
                galaxy_dict[key] = []
            try:
                galaxy_dict[key].append(
                    df[key][(df.idx == idx) & (df.snap == snap)].values[0]
                )
            except:
                galaxy_dict[key].append(np.nan)
    return galaxy_dict


def load_sfr_hist(snap):
    sfr_hist_path = config["sfr_hist_path"]
    if snap < 10:
        snap_num = "00" + str(snap)
    elif snap < 100:
        snap_num = "0" + str(snap)
    file_name = f"Subhalo_SFRs_{snap_num}.hdf5"
    file_path = os.path.join(sfr_hist_path, file_name)
    with h5py.File(file_path, "r") as f:
        sfr_hist10 = np.array(f["Subhalo"]["SFR_MsunPerYrs_in_all_10Myrs"])
        sfr_hist50 = np.array(f["Subhalo"]["SFR_MsunPerYrs_in_all_50Myrs"])
        sfr_hist100 = np.array(f["Subhalo"]["SFR_MsunPerYrs_in_all_100Myrs"])
    return sfr_hist10, sfr_hist50, sfr_hist100


def update_sfr_hist(galaxy_dict):
    SFR_hist10 = []
    SFR_hist50 = []
    SFR_hist100 = []
    for snap, idx in zip(galaxy_dict["snap"], galaxy_dict["idx"]):
        if np.isnan(idx):
            SFR_hist10.append(np.nan)
            SFR_hist50.append(np.nan)
            SFR_hist100.append(np.nan)
            continue
        else:
            sfr_hist10, sfr_hist50, sfr_hist100 = load_sfr_hist(snap)
            SFR_hist10.append(sfr_hist10[idx])
            SFR_hist50.append(sfr_hist50[idx])
            SFR_hist100.append(sfr_hist100[idx])
    galaxy_dict["SFR_hist10"] = SFR_hist10
    galaxy_dict["SFR_hist50"] = SFR_hist50
    galaxy_dict["SFR_hist100"] = SFR_hist100
    return


def outflow_props(df, halo_id, snap, aperture_size):
    gal = Galaxy(
        df=df,
        halo_id=halo_id,
        snap=snap,
        aperture_size=aperture_size,
    )
    keys = ["M_out_0.6", "M_out", "M_dot"]
    out_props = {}
    a = gal.get_outflow_mass(in_aperture=True)

    try:
        out_props["M_out_0.6"] = gal.get_outflow_mass(in_aperture=True)
        out_props["M_out"] = gal.get_outflow_mass(in_aperture=False)
        out_props["M_dot"] = gal.get_flow_rate(in_aperture=True)
    except:
        for key in keys:
            out_props[key] = np.nan
    return out_props


def update_outflow_props(df, galaxy_dicts, aperture_size):
    for gal, galaxy_dict in galaxy_dicts.items():
        print(f"Working on galaxy {gal}")
        galaxy_dict["M_out_0.6"] = []
        galaxy_dict["M_out"] = []
        galaxy_dict["M_dot"] = []

        for snap, idx in zip(galaxy_dict["snap"], galaxy_dict["idx"]):
            if np.isnan(idx):
                galaxy_dict["M_out_0.6"].append(np.nan)
                galaxy_dict["M_out"].append(np.nan)
                galaxy_dict["M_dot"].append(np.nan)
                continue
            else:
                halo_id = df[(df.idx == idx) & (df.snap == snap)].Halo_id.values[0]
                out_props = outflow_props(
                    df, halo_id, snap, aperture_size=aperture_size
                )
                for key, value in out_props.items():
                    galaxy_dict[key].append(value)
    return


def add_time(galaxies):
    for galaxy_dict in galaxies.values():
        galaxy_dict["z"] = []
        galaxy_dict["time"] = []
        galaxy_dict["lookback"] = []
        for snap in galaxy_dict["snap"]:
            if np.isnan(snap):
                galaxy_dict["z"].append(np.nan)
                galaxy_dict["time"].append(np.nan)
                galaxy_dict["lookback"].append(np.nan)

                continue
            else:
                z = get_redshift(snap)
                time = TNGcosmo.age(z).value
                galaxy_dict["z"].append(z)
                galaxy_dict["time"].append(time)
                galaxy_dict["lookback"].append(TNGcosmo.lookback_time(z).value)
    return


def add_log_quantities(galaxies):
    for galaxy_dict in galaxies.values():
        keys = list(galaxy_dict.keys())
        for key in keys:
            if key in [
                "M_out",
                "M_out_0.6",
            ]:
                galaxy_dict[key + "_log"] = np.log10(
                    np.array(galaxy_dict[key]).astype(float) * 1e10 / 0.6774
                )
            elif key in [
                "Galaxy_SFR",
                "SFR_hist10",
                "SFR_hist50",
                "SFR_hist100",
            ]:
                galaxy_dict[key + "_log"] = np.log10(
                    np.array(galaxy_dict[key]).astype(float)
                )
    return


def add_sSFR(galaxies):
    for galaxy_dict in galaxies.values():
        galaxy_dict["sSFR_log"] = np.array(galaxy_dict["SFR_hist10_log"]) - np.array(
            galaxy_dict["M_star_log"]
        )
        galaxy_dict["sSFR_log_100"] = np.array(
            galaxy_dict["SFR_hist100_log"]
        ) - np.array(galaxy_dict["M_star_log"])
        galaxy_dict["sOutflow"] = np.array(galaxy_dict["M_out_log"]) - np.array(
            galaxy_dict["M_star_log"]
        )
    return


def add_mass_loading(galaxies):
    for galaxy_dict in galaxies.values():
        galaxy_dict["eta_log"] = np.array(galaxy_dict["M_dot_log"]) - np.array(
            galaxy_dict["SFR_hist10_log"]
        )
    return


def add_bh_growth(galaxies):
    for galaxy_dict in galaxies.values():
        galaxy_dict["BH_mdot_log"] = np.log10(
            np.array(galaxy_dict["BH_growth"]) * 1e10 / 0.978
        )
    return


def convert_outflow_rate(galaxies):
    for galaxy_dict in galaxies.values():
        s_to_yr = 31557600.0
        kpc_to_km = 3.086e16
        galaxy_dict["M_dot_conv"] = np.log10(
            np.array(galaxy_dict["M_dot"]) * 1e10 / 0.6774 * s_to_yr / kpc_to_km
        )
        galaxy_dict["M_dot_log"] = np.log10(galaxy_dict["M_dot_conv"] * 1e6)
    return


def get_progenitor_histories(df, idces, snaps=None):
    if snaps is None:
        snaps = [25] * len(idces)
    galaxies = {}
    for idx, snap in zip(idces, snaps):
        galaxy_dict = get_progenitor_history_dict(
            df=df,
            galaxy_idx=idx,
            snap_num=snap,
            merger_history_path=config["merger_history_path"],
        )
        update_sfr_hist(galaxy_dict)
        galaxies[idx] = galaxy_dict
    update_outflow_props(df, galaxies, aperture_size=0.6)
    add_time(galaxies)
    add_log_quantities(galaxies)
    add_sSFR(galaxies)
    convert_outflow_rate(galaxies)
    add_mass_loading(galaxies)
    add_bh_growth(galaxies)
    return galaxies


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
