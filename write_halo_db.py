import os
import h5py
from config import config
from utils import get_halo_data


class GalaxyWriter:

    def __init__(self, galaxy, save_name, data_of_interest=None):
        self.galaxy = galaxy
        self.save_name = save_name

        self._data_dict = None
        self._data_of_interest = data_of_interest

    @property
    def data_of_interest(self):
        if self._data_of_interest is None:
            self._data_of_interest = {"full_galaxy", "out_galaxy", "out_gas"}
        return self._data_of_interest

    @property
    def data_dict(self):
        if self._data_dict is None:
            self._data_dict = {}

            halo_info = get_halo_data(
                self.galaxy.df, self.galaxy.halo_id, self.galaxy.snap
            )

            self._data_dict["info"] = halo_info
            if "full_galaxy" in self.data_of_interest:
                self._data_dict["full_galaxy"] = self._select_keys_of_interest(
                    self.galaxy.gas
                )
            if "out_galaxy" in self.data_of_interest:
                self._data_dict["out_galaxy"] = self._select_keys_of_interest(
                    self.galaxy.out_galaxy
                )
            if "out_gas" in self.data_of_interest:
                self._data_dict["out_gas"] = self._select_keys_of_interest(
                    self.galaxy.out_gas
                )
        return self._data_dict

    def _select_keys_of_interest(self, gas):
        keys = [
            "Coordinates",
            "Velocities",
            "Masses",
            "GFM_Metallicity",
            "Temperature",
            "StarFormationRate",
            "Flow_Velocities",
            "hsml",
        ]
        new_gas = {}
        for key, value in gas.items():
            if key in keys:
                new_gas[key] = value
        return new_gas

    def save_to_db(self):
        file_path = os.path.join(
            config["base_path"], self.save_name + config["hdf_ending"]
        )
        with h5py.File(file_path, "a") as hdf_file:
            snap_group = self._create_or_open_group(hdf_file)
            gal_group_name = str(self.galaxy.halo_id)
            if gal_group_name in snap_group:
                del snap_group[gal_group_name]
            galaxy_group = snap_group.create_group(gal_group_name)
            for key, value in self.data_dict.items():
                if key == "info":
                    for info_key, info_value in self.data_dict[key].items():
                        galaxy_group.attrs[info_key] = info_value
                else:
                    subgroup = galaxy_group.create_group(key)

                    for subkey, subvalue in value.items():
                        subgroup[subkey] = subvalue
        return

    def _create_or_open_group(self, hdf_file):
        group_name = f"snap_{self.galaxy.snap}"
        if group_name in hdf_file:
            return hdf_file[group_name]
        else:
            return hdf_file.create_group(group_name)


## Keep this here to add the history writing later on
# def write_halo_db(
#     df,
#     halo_id,
#     snap,
#     zoom_in="autozoom",
#     n_peak=4,
#     group_props=None,
#     with_history=False,
# ):

#     if with_history:
#         galaxy_idx = get_galaxyID_from_haloID(
#             df=df, halo_id=halo_id, snap=snap
#         )
#         galaxy_idces, snap_nums = get_progenitor_history(
#             galaxy_idx=galaxy_idx, snap_num=snap
#         )
#         full_data_dict = {}
#         for idx, snap_num in zip(galaxy_idces, snap_nums):
#             try:
#                 halo_idx = get_haloID_from_galaxyID(
#                     df=df, galaxy_id=idx, snap=snap_num
#                 )
#             except IndexError:
#                 break
#             full_data_dict[snap_num] = create_halo_dict(
#                 df,
#                 halo_idx,
#                 snap_num,
#                 zoom_in=zoom_in,
#                 n_peak=n_peak,
#                 group_props=group_props,
#             )

#     else:
#         full_data_dict = create_halo_dict(
#             df,
#             halo_id,
#             snap,
#             zoom_in=zoom_in,
#             n_peak=n_peak,
#             group_props=group_props,
#         )
#     if with_history:
#         filename = f"{halo_id}_history.pickle"
#     else:
#         filename = f"{halo_id}.pickle"
#     file_path = os.path.join(
#         config["base_path"], config["dir_prefix"] + str(snap), filename
#     )
#     filehandler = open(file_path, "wb")
#     pickle.dump(full_data_dict, filehandler)
#     return
