import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from utils import map_to_new_dict
from gaussian_outflow_selection import (
    select_galaxy_group,
    group_gas,
    get_only_outflowing_gas,
)
from plotting import label_colors


def load_test_gal():
    hdf5_file_path = "/ptmp/mpa/ivkos/outflows/model.h5"
    with h5py.File(hdf5_file_path, "r") as hdf5_file:
        galaxy_np = hdf5_file["model"][:]
    return galaxy_np


def create_galaxy_dict(galaxy_np):
    galaxy = {}
    galaxy["out_id"] = galaxy_np[0, :]
    galaxy["Coordinates"] = galaxy_np[1:4, :].T
    galaxy["idces"] = np.arange(galaxy["Coordinates"].shape[0])
    galaxy["Velocities"] = galaxy_np[4:7, :].T
    galaxy["StarFormationRate"] = galaxy_np[7, :]
    return galaxy


def create_galaxy_extended_dict(galaxy_np):
    galaxy = {}
    galaxy["out_id"] = galaxy_np[0, :]
    galaxy["Coordinates"] = galaxy_np[1:4, :].T
    galaxy["idces"] = np.arange(galaxy["Coordinates"].shape[0])
    galaxy["Velocities"] = galaxy_np[4:7, :].T
    galaxy["StarFormationRate"] = galaxy_np[7, :]
    galaxy["Out_Coordinates"] = galaxy["Coordinates"][galaxy["out_id"] == 1, :]
    galaxy["Out_Velocities"] = galaxy["Velocities"][galaxy["out_id"] == 1, :]
    galaxy["Out_SFR"] = galaxy["StarFormationRate"][galaxy["out_id"] == 1]
    galaxy["out_idces"] = galaxy["idces"][galaxy["out_id"] == 1]
    galaxy["In_Coordinates"] = galaxy["Coordinates"][galaxy["out_id"] == 0, :]
    galaxy["In_Velocities"] = galaxy["Velocities"][galaxy["out_id"] == 0, :]
    galaxy["In_SFR"] = galaxy["StarFormationRate"][galaxy["out_id"] == 0]
    galaxy["in_idces"] = galaxy["idces"][galaxy["out_id"] == 0]
    return galaxy


def select_moving_gas(gas, threshold_velocity=0):
    idces = gas["Flow_Velocities"] > threshold_velocity
    return map_to_new_dict(gas, idces)


def select_gas_group(gas, group_num):
    idces = gas["group"] == group_num
    return map_to_new_dict(gas, idces)


def get_gas_groups(gas):
    n_groups = np.max(gas["group"]) + 1
    return [select_gas_group(gas, i) for i in range(n_groups)]


def identify_galaxy_group(gas_groups, test=True):
    group_num = select_galaxy_group(gas_groups, use_weighted_distance=False, test=test)
    galaxy_group = gas_groups[group_num]
    return group_num, galaxy_group


def get_out_gas(galaxy):
    norm = np.linalg.norm(galaxy["Coordinates"], axis=1)[:, np.newaxis]
    galaxy["Relative_Distances"] = np.linalg.norm(
        galaxy["Coordinates"], axis=1
    )
    galaxy["Direction"] = galaxy["Coordinates"] / norm
    galaxy["Flow_Velocities"] = np.float32(
        (galaxy["Velocities"] * galaxy["Direction"]).sum(axis=1)
    )
    all_out_gas = select_moving_gas(galaxy, threshold_velocity=0)
    group_gas(all_out_gas, props=None, n_peak=3)
    gas_groups = get_gas_groups(all_out_gas)
    galaxy_group_num, out_galaxy = identify_galaxy_group(gas_groups, test=True)
    outflowing_gas = get_only_outflowing_gas(
        all_out_gas,
        galaxy_group=galaxy_group_num,
        crit_vout=1e5,
    )
    return outflowing_gas


def identify_out_gas():
    galaxy_np = load_test_gal()
    galaxy = create_galaxy_dict(galaxy_np)
    out_gas = get_out_gas(galaxy)
    return out_gas


def compare_out_selection():
    out_gas_gmm = identify_out_gas()
    galaxy_ext = create_galaxy_extended_dict(load_test_gal())
    out_idces_gmm = set(out_gas_gmm["idces"].tolist())
    out_idces_true = set(galaxy_ext["out_idces"].tolist())
    missing_idces = out_idces_true - out_idces_gmm
    number_missing = len(missing_idces)
    extra_idces = out_idces_gmm - out_idces_true
    number_extra = len(extra_idces)
    print(f"Number of missing outflowing gas particles: {number_missing}")
    print(f"Number of extra outflowing gas particles: {number_extra}")
    print(f"Fraction of missing particles: {number_missing / len(out_idces_true):.4f}")
    print(f"Fraction of extra particles: {number_extra / len(out_idces_true):.4f}")


def plot_outflow_comparison_test(for_slides=False):
    label_colors(for_slides)
    out_gas = identify_out_gas()
    galaxy = create_galaxy_extended_dict(load_test_gal())
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(7, 13))
    ax0, ax1, ax2 = axes

    plt.subplots_adjust(hspace=0)

    all_y = np.concatenate(
        [
            galaxy["Coordinates"][:, 2],
            out_gas["Coordinates"][:, 2],
            galaxy["Out_Coordinates"][:, 2],
        ]
    )
    ymin, ymax = np.min(all_y), np.max(all_y)
    yrange = ymax - ymin
    ymin -= 0.1 * yrange
    ymax += 0.1 * yrange

    if for_slides:
        cmap = ListedColormap(["#00FFFF", "#FFA500"])
    else:
        cmap = plt.get_cmap("coolwarm")

    ax0.scatter(
        galaxy["Coordinates"][:, 1],
        galaxy["Coordinates"][:, 2],
        s=0.1,
        c=galaxy["out_id"],
        cmap=cmap,
    )
    ax0.text(0.98, 0.65, "full galaxy", transform=ax0.transAxes,
             ha="right", va="top", fontsize=16)
    ax0.set_ylabel("z", fontsize=20)
    ax0.set_ylim(ymin, ymax)
    ax0.tick_params(axis="both", length=5, labelsize=15)
    ax0.tick_params(axis="x", labelbottom=False)

    ax1.scatter(
        out_gas["Coordinates"][:, 1],
        out_gas["Coordinates"][:, 2],
        s=0.1,
        c=out_gas["out_id"],
        cmap=cmap,
    )
    ax1.text(0.98, 0.65, "identified outflows", transform=ax1.transAxes,
             ha="right", va="top", fontsize=16)
    ax1.set_ylabel("z", fontsize=20)
    ax1.set_ylim(ymin, ymax)
    ax1.tick_params(axis="both", length=5, labelsize=15)
    ax1.tick_params(axis="x", labelbottom=False)

    red_match = cmap(1.0)
    ax2.scatter(
        galaxy["Out_Coordinates"][:, 1],
        galaxy["Out_Coordinates"][:, 2],
        s=0.1,
        c=red_match,
    )
    ax2.text(0.98, 0.65, "actual outflows", transform=ax2.transAxes,
             ha="right", va="top", fontsize=16)
    ax2.set_xlabel("x", fontsize=20)
    ax2.set_ylabel("z", fontsize=20)
    ax2.set_ylim(ymin, ymax)
    ax2.tick_params(axis="both", length=5, labelsize=15)
    plt.savefig("test.png")
