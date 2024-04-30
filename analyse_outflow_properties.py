import numpy as np
from utils import (
    get_halo,
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


def filter_out_gas(gas, n_peak, group_props=None):
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
    return out_gas


def outflow_mass_vout(
    df,
    halo_id,
    snap,
    zoom_in="autozoom",
    n_peak=4,
    group_props=None,
    zoom_in_factor=20,
):

    gas = retrieve_halo_gas(df=df, snap=snap, halo_id=halo_id)

    halo = get_halo(df=df, snap=snap, halo_id=halo_id)
    r_vir = float(halo.R_vir)

    if zoom_in == "autozoom":
        zoom_in = autozoom(halo.R_vir, halo.Galaxy_HMR, factor=zoom_in_factor)
        print(zoom_in)
    gas = cut_zoomed(gas=gas, r_vir=r_vir, zoom_in=zoom_in)

    out_gas = filter_out_gas(gas, n_peak=n_peak, group_props=group_props)
    M_out = out_gas["Masses"].sum()
    v_out_mean = np.average(
        out_gas["Flow_Velocities"], weights=out_gas["Masses"]
    )
    return v_out_mean, M_out


def add_outflow_parameters(df, snap=None):
    if "v_out_mean" not in df.keys():
        df["v_out_mean"] = np.nan * np.ones(len(df))
    if "M_out" not in df.keys():
        df["M_out"] = np.nan * np.ones(len(df))

    if snap is not None:
        iteration_df = df[df.snap == snap]
    else:
        iteration_df = df
    for _, element in iteration_df:
        halo_id = int(element.Halo_id)
        snap = element.snap
        v_out_mean, M_out = outflow_mass_vout(
            df,
            halo_id,
            snap,
            zoom_in="autozoom",
            n_peak=4,
            group_props=None,
            zoom_in_factor=20,
        )
        df.loc[(df.snap == snap) & (df.Halo_id == halo_id)][
            "v_out_mean"
        ] = v_out_mean
        df.loc[(df.snap == snap) & (df.Halo_id == halo_id)]["M_out"] = M_out
    return
