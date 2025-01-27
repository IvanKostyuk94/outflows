import os
import numpy as np
import pandas as pd
from config import config
from utils import sort_all_keys
from process_gas import Galaxy
import h5py


class SFR_hist_updater:
    def __init__(self, df_name, save_name=None):
        self.df_name = df_name
        self.save_name = save_name

    def set_paths(self):
        base_path = config["base_path"]
        self.df_path = os.path.join(base_path, self.df_name)
        if self.save_name is None:
            self.save_path = self.df_path
        else:
            self.save_path = os.path.join(base_path, self.save_name)
        return

    def load_df(self):
        self.df = pd.read_hdf(self.df_path)
        return
    
    def save_df(self):
        self.df.to_hdf(self.save_path, config["hdf_key"])
        return

    def load_sfr_hist(self, snap):
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

    def update_sfr_hist(self):
        self.df["SFR_hist10"] = np.nan * np.ones(len(self.df))
        self.df["SFR_hist50"] = np.nan * np.ones(len(self.df))
        self.df["SFR_hist100"] = np.nan * np.ones(len(self.df))
        for snap in self.df.snap.unique():
            print(f'Working on snapshot {snap}')
            sub_df = self.df.loc[self.df.snap == snap]
            sfr_hist10, sfr_hist50, sfr_hist100 = self.load_sfr_hist(snap)
            for _, element in sub_df.iterrows():
                idx = int(element.idx)
                self.df.loc[
                    (self.df.snap == snap) & (self.df.idx == idx), "SFR_hist10"
                ] = sfr_hist10[idx]
                self.df.loc[
                    (self.df.snap == snap) & (self.df.idx == idx), "SFR_hist50"
                ] = sfr_hist50[idx]
                self.df.loc[
                    (self.df.snap == snap) & (self.df.idx == idx), "SFR_hist100"
                ] = sfr_hist100[idx]
        return

    def update_df(self):
        self.set_paths()
        self.load_df()
        self.update_sfr_hist()
        self.save_df()
        return
    
if __name__ == "__main__":
    df_name = "in_aperture_06sec_v_90_W80_newMout.hdf5"
    save_name = "in_aperture_06sec_v_90_W80_newMout_SFR.hdf5"
    sfr_hist_updater = SFR_hist_updater(df_name, save_name)
    sfr_hist_updater.update_df()