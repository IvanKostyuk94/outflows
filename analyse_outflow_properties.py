import os
import numpy as np
import pandas as pd
from config import config
from process_gas import Galaxy


class OutflowPropUpdater:
    def __init__(
        self, df_name, snap=None, save_name=None, n_peak=3, group_props=None
    ):
        self.df_name = df_name + config["hdf_ending"]
        self.snap = snap
        self.save_name = save_name

        self.n_peak = n_peak
        self.group_props = group_props

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
                self._save_path = os.path.join(base_path, self.save_name)
        return self._save_path

    @property
    def df(self):
        if self._df is None:
            self._df = pd.read_hdf(self.df_path)
        return self._df

    def outflow_props(self, halo_id, snap):
        gal = Galaxy(
            df=self.df,
            halo_id=halo_id,
            snap=snap,
            n_peak=self.n_peak,
            group_props=self.group_props,
        )
        out_props = {}
        out_props["M_out"] = gal.get_outflow_mass()
        out_props["M_dot"] = gal.get_flow_rate()
        out_props["v_lum"] = gal.get_average_outflow_vel(
            weighting="Luminosity"
        )
        out_props["v_mass"] = gal.get_average_outflow_vel(weighting="Masses")
        out_props["M_out_cold"] = gal.get_outflow_mass(cold_only=True)
        out_props["M_dot_cold"] = gal.get_flow_rate(cold_only=True)
        out_props["v_lum_cold"] = gal.get_average_outflow_vel(
            weighting="Luminosity", cold_only=True
        )
        out_props["v_mass_cold"] = gal.get_average_outflow_vel(
            weighting="Masses", cold_only=True
        )
        return out_props

    def create_key_column(self, key):
        if key not in self.df.keys():
            self.df[key] = np.nan * np.ones(len(self.df))
        return

    def add_outflow_parameters(self):
        if "v_out_mean" not in self.df.keys():
            self.df["v_out_mean"] = np.nan * np.ones(len(self.df))
        if "M_out" not in self.df.keys():
            self.df["M_out"] = np.nan * np.ones(len(self.df))
        if self.snap is not None:
            iteration_df = self.df[self.df.snap == self.snap]
        else:
            iteration_df = self.df
        for _, element in iteration_df.iterrows():
            halo_id = int(element.Halo_id)
            snap = int(element.snap)
            out_props = self.outflow_props(halo_id, snap)
            for key in out_props.keys():
                self.create_key_column(key)
                self.df.loc[
                    (self.df.snap == snap) & (self.df.Halo_id == halo_id), key
                ] = out_props[key]
        return

    def save_df(self):
        self.df.to_hdf(self.save_path, "galaxies")


if __name__ == "__main__":
    updater = OutflowPropUpdater("all_galaxies_new")
    updater.add_outflow_parameters()
    updater.save_df()
