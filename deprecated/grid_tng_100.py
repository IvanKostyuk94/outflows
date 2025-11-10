import utils
import numpy as np
from pyTNG import gas_temperature
from pyTNG import gridding
from pyTNG.utils import dfFromArrDict


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
    dataset_df = dfFromArrDict(sub_dict)
    return dataset_df


def get_hsml(gas):
    gas["hsml"] = 2.5 * (
        3 * (gas["Masses"] / gas["Density"]) / (4.0 * np.pi)
    ) ** (1.0 / 3)
    return


def convert_gas(size=100):
    _, sim_path = get_sim()
    galaxy_id = 490053
    snap = 99
    galaxy_center = np.array([45814.31640625, 58883.9375, 34564.125])
    gas = il.snapshot.loadSubhalo(sim_path, snap, galaxy_id, "gas")
    convert_distance = 1 / 0.6774
    convert_mass = 1e10 / 0.6774
    convert_density = (1e10 / 0.6774) / (1 / 0.6774) ** 3

    convert_electron_density = 1.989e33 / 1.67e-24 * 0.76 / 3.086e21**3
    convert_gauss = 0.6774 * 2.6e-6
    grid_center = np.array([0, 0, 0])
    gas["Coordinates"] = gas["Coordinates"] - galaxy_center
    gas["Coordinates"] = gas["Coordinates"] * convert_distance
    gas["Masses"] = gas["Masses"] * convert_mass
    gas["Density"] = gas["Density"] * convert_density
    get_hsml(gas)
    gas["MagneticField_x"] = gas["MagneticField"][:, 0] * convert_gauss
    gas["MagneticField_y"] = gas["MagneticField"][:, 1] * convert_gauss
    gas["MagneticField_z"] = gas["MagneticField"][:, 2] * convert_gauss
    boxsize = 200
    gas_temperature.gasTemp(gas)
    gas["electron_density"] = (
        gas["Density"] * gas["ElectronAbundance"] * convert_electron_density
    )
    quants = ["electron_density", "Temperature"]
    grids = gridding.depositParticlesOnGrid(
        gas_parts=gas,
        method="sphKernelDep",
        quants=quants,
        box_size_parts=0 * np.ones(3),
        grid_shape=(size * np.ones(3)).astype(np.int64),
        grid_size=boxsize * np.ones(3),
        grid_cen=grid_center,
        n_threads=8,
    )
    for quant in quants:
        grids[quant] = np.where(
            grids["Masses"] != 0, grids[quant] / grids["Masses"], 0
        )
    for dir in ["x", "y", "z"]:
        grid = gridding.depositParticlesOnGrid(
            gas_parts=gas,
            method="sphKernelDep",
            quants=[],
            box_size_parts=2 * 0 * np.ones(3),
            grid_shape=(size * np.ones(3)).astype(np.int64),
            grid_size=boxsize * np.ones(3),
            grid_cen=grid_center,
            n_threads=8,
            mass_key=f"MagneticField_{dir}",
        )
        grids[f"MagneticField_{dir}"] = grid[f"MagneticField_{dir}"]
    return grids


def save_dict_to_hdf5(data_dict, file_path=None):
    if file_path is None:
        file_path = "/u/ivkos/490053.hdf5"
    with h5py.File(file_path, "w") as hdf_file:
        for key, value in data_dict.items():
            hdf_file.create_dataset(key, data=value)
    return