import os
import numpy as np
import pandas as pd
from config import config
from los_projection import GalaxyProjections

class RandomProjection:
    def __init__(
        self,
        df_name,
        save_name=None,
        snap_range=None,
        in_aperture=True,
        aperture_size=0.6,
        serra=False,
    ):
        self.df_name = df_name + config["hdf_ending"]
        self.save_name = save_name
        self.snap_range = snap_range

        self.in_aperture = in_aperture
        self.aperture_size = aperture_size
        self.serra = serra

        self._df_path = None
        self._save_path = None
        self._df = None

    @property
    def df_path(self):
        if self._df_path is None:
            base_path = config["base_path"]
            self._df_path = os.path.join(base_path, self.df_name)
        return self._df_path
    
    @property
    def save_path(self):
        if self._save_path is None:
            base_path = config["base_path"]
            if self.save_name is None:
                self._save_path = self.df_path
            else:
                self._save_path = os.path.join(
                    base_path, self.save_name + config["hdf_ending"]
                )
        return self._save_path

    @property
    def df(self):
        if self._df is None:
            self._df = pd.read_hdf(self.df_path)
            self._df = self._df[
                (self.df.snap >= self.snap_range[0])
                & (self.df.snap <= self.snap_range[1])
                & (self.df.M_star_log > 7.5)
                & (self.df.M_star_log < 8.5)
            ].copy(deep=True)

        return self._df
    
    def get_offset_W80(self, gas):
        try:
            v_range = np.quantile(gas["los_Velocities"], [0.1, 0.9])
            W80 = v_range[1] - v_range[0]
        except:
            W80 = np.nan
        return W80
    
    def add_random_W80(self):
        iteration_df = self.df
        counter = 0
        phi = 0
        u = np.random.default_rng().uniform(0.0, 1.0, len(iteration_df)) 
        theta_angles = np.degrees(np.arccos(u))
        keys = [
            "W80_sample_outflow_aperture",
            "W80_sample_galaxy_aperture",
        ]
        for key in keys:
            self.df[key] = np.nan * np.ones(
                                len(self.df)
                            )
        print("Starting random projections")
            
        for _, element in iteration_df.iterrows():
            theta = theta_angles[counter]
            counter += 1
            if self.serra:
                halo_id = int(element.idx)
            else:
                halo_id = int(element.Halo_id)
            snap = int(element.snap)

            gal = GalaxyProjections(
                df=self.df,
                halo_id=halo_id,
                snap=snap,
                projection_angle_theta=theta,
                projection_angle_phi=phi,
                aperture_size=self.aperture_size,
                serra=self.serra,
            )
            try:
                gal.project_outflows()
                if self.in_aperture:
                    remain_gas = gal.get_in_aperture(gal.remain_gas)
                    out_gas = gal.get_in_aperture(gal.out_gas)
                else:
                    remain_gas = gal.remain_gas
                    out_gas = gal.out_gas

                W80_galaxy = self.get_offset_W80(remain_gas)
                W80_outflow = self.get_offset_W80(out_gas)
            except:
                W80_galaxy = np.nan
                W80_outflow = np.nan

            self.df.loc[
                (self.df.snap == snap) & (self.df.Halo_id == halo_id),
                "W80_sample_galaxy_aperture",
            ] = W80_galaxy

            self.df.loc[
                (self.df.snap == snap) & (self.df.Halo_id == halo_id),
                "W80_sample_outflow_aperture",
            ] = W80_outflow
            if counter % 100 == 0:
                print(f"Processed {counter/len(iteration_df)*100:.2f}% of galaxies")
                self.save_df()
        return
        
    def save_df(self):
        self.df.to_hdf(self.save_path, key="galaxies")

if __name__ == "__main__":
    updater = RandomProjection(df_name="in_aperture_final", 
                               save_name="random_projections", 
                               snap_range=(13, 26), 
                               in_aperture=True, 
                               aperture_size=0.6, 
                               serra=False)
    updater.add_random_W80()
    updater.save_df()

