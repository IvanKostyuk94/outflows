import os
import numpy as np
import pandas as pd
from config import config
from utils import sort_all_keys
from process_gas import Galaxy
import h5py


class Wind_mass_updater:
    def __init__(self, df_name, save_name=None):
        self.df_name = df_name
        self.save_name = save_name
        self.h = 0.6774

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
    
    def get_wind_mass(self, element):
        gal = Galaxy(df=self.df, halo_id=int(element.Halo_id), 
                 snap=int(element.snap), 
                 aperture_size=0.6)
        wind_mass = np.sum(gal.wind['Masses'])*1e10/self.h
        return wind_mass


    def update_wind_mass(self):
        counter = 0
        self.df["Mass_wind_particles"] = np.nan * np.ones(len(self.df))
        for _, element in self.df.iterrows():
            idx = int(element.idx)
            self.df.loc[
                (self.df.snap == element.snap) & (self.df.idx == element.idx), "Mass_wind_particles"
            ] = self.get_wind_mass(element)
            counter += 1
            if counter % 1000 == 0:
                print(f"Processed {counter/len(self.df)}% of galaxies")
        return

    def update_df(self):
        self.set_paths()
        self.load_df()
        self.update_wind_mass()
        self.save_df()
        return
    
if __name__ == "__main__":
    df_name = "in_aperture_final.hdf5"
    save_name = "in_aperture_with_wind.hdf5"
    wind_mass_updater = Wind_mass_updater(df_name, save_name)
    wind_mass_updater.update_df()