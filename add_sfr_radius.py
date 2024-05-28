import os
import numpy as np
import pandas as pd
from config import config
from utils import sort_all_keys
from process_gas import Galaxy


class R_SFR_Updater:

    def __init__(self, df_name, snap=None, save_name=None):
        self.df_name = df_name
        self.snap = snap
        self.save_name = save_name
        self.set_paths()
        self.load_df()

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

    def compute_sfr_radius(self, particles, tot_sfr):
        reduced_particles = {}
        relevant_keys = ["Relative_Distances", "StarFormationRate"]
        for key in relevant_keys:
            reduced_particles[key] = particles[key]
        sort_all_keys(
            particles=reduced_particles, sort_key="Relative_Distances"
        )
        sfr_cum = np.cumsum(reduced_particles["StarFormationRate"])
        index_SFR = np.searchsorted(sfr_cum, tot_sfr / 2, side="right")
        try:
            r_SFR = reduced_particles["Relative_Distances"][index_SFR]
        except IndexError:
            r_SFR = np.nan
        return r_SFR

    def get_sfr_radius(self, gal):
        gal_sfr = float(gal.halo.Galaxy_SFR)
        gal.retrieve_halo_gas()
        r_SFR = self.compute_sfr_radius(particles=gal.gas, tot_sfr=gal_sfr)
        return r_SFR

    def add_sfr_radius_column(self):
        if "r_SFR" not in self.df.keys():
            self.df["r_SFR"] = np.nan * np.ones(len(self.df))
        if self.snap is not None:
            iteration_df = self.df[self.df.snap == self.snap]
        else:
            iteration_df = self.df
        counter = 0
        for _, element in iteration_df.iterrows():
            halo_id = int(element.Halo_id)
            snap = int(element.snap)
            gal = Galaxy(df=self.df, snap=snap, halo_id=halo_id)
            if np.isnan(gal.halo.r_SFR.values[0]):
                r_SFR = self.get_sfr_radius(gal)
                if r_SFR == np.nan:
                    print(
                        f"No radius was found for halo {halo_id} in snap {snap}"
                    )
                self.df.loc[
                    (self.df.snap == snap) & (self.df.Halo_id == halo_id),
                    "r_SFR",
                ] = r_SFR
            if counter % 100 == 0:
                print(f"Processed {counter/len(self.df)*100:.2f}% of galaxies")
            counter += 1
            if counter % 1000 == 0:
                self.save_df()
        return

    def update_df(self):
        self.add_sfr_radius_column()
        self.save_df()


if __name__ == "__main__":
    updater = R_SFR_Updater(df_name="all_galaxies_new.hdf5")
    updater.update_df()
