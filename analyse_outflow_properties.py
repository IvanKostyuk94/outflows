import os
import numpy as np
import pandas as pd
from config import config
from process_gas import Galaxy
from galaxy_shell_outflows import GalaxyShells
from los_projection import GalaxyProjections


class OutflowPropUpdater:
    def __init__(
        self,
        df_name,
        save_name=None,
        group_props=None,
        snap_range=None,
        with_quantile=False,
        only_shell=False,
    ):
        self.df_name = df_name + config["hdf_ending"]
        self.snap_range = snap_range
        self.save_name = save_name
        self.shell = only_shell

        self.group_props = group_props
        self.with_quantile = with_quantile

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
            ].copy(deep=True)

        return self._df

    def outflow_props(self, halo_id, snap):
        gal = Galaxy(
            df=self.df,
            halo_id=halo_id,
            snap=snap,
            group_props=self.group_props,
        )
        keys = [
            "M_out",
            "M_dot",
            "v_lum",
            "v_mass",
            "M_out_cold",
            "M_dot_cold",
            "v_lum_cold",
            "v_mass_cold",
        ]
        out_props = {}
        try:
            out_props["M_out"] = gal.get_outflow_mass()
            out_props["M_dot"] = gal.get_flow_rate()
            out_props["v_lum"] = gal.get_average_outflow_vel(weighting="Luminosity")
            out_props["v_mass"] = gal.get_average_outflow_vel(weighting="Masses")
            out_props["M_out_cold"] = gal.get_outflow_mass(cold_only=True)
            out_props["M_dot_cold"] = gal.get_flow_rate(cold_only=True)
            out_props["v_lum_cold"] = gal.get_average_outflow_vel(
                weighting="Luminosity", cold_only=True
            )
            out_props["v_mass_cold"] = gal.get_average_outflow_vel(
                weighting="Masses", cold_only=True
            )
        except:
            for key in keys:
                out_props[key] = None
        return out_props

    def quantile_outflow_props(self, halo_id, snap):
        if self.shell:
            # This is mostly to compare with Nelson et al. 2019,
            # hence the hardcoded 10kpc radius
            gal = GalaxyShells(
                df=self.df,
                halo_id=halo_id,
                snap=snap,
                radius=10,
            )
        else:
            gal = Galaxy(
                df=self.df,
                halo_id=halo_id,
                snap=snap,
                group_props=self.group_props,
            )
        keys = []
        out_vels = {}
        quantiles = [0.5, 0.75, 0.9]
        weightings = ["Luminosity", "Masses", "Flux"]
        subscripts = ["lum", "mass", "mdot"]
        temps = ["", "_cold"]
        for sub, weighting in zip(subscripts, weightings):
            for quantile in quantiles:
                for i, temp in enumerate(temps):
                    column_name = f"v_{sub}_{int(quantile*100)}{temp}"
                    keys.append(column_name)
        try:
            for sub, weighting in zip(subscripts, weightings):
                for quantile in quantiles:
                    for i, temp in enumerate(temps):
                        column_name = f"v_{sub}_{int(quantile*100)}{temp}"
                        out_vels[column_name] = gal.get_quantile_velocity(
                            quantile, weighting=weighting, cold_only=bool(i)
                        )
        except:
            for key in keys:
                out_vels[key] = None

        return out_vels

    def _create_key_column(self, key):
        if key not in self.df.keys():
            self.df[key] = np.nan * np.ones(len(self.df))
        return

    def _has_value(self, halo_id, snap):
        try:
            if self.with_quantile:
                column = "v_lum_50"
            else:
                column = "M_out"
            value = self.df.loc[
                (self.df.snap == snap) & (self.df.Halo_id == halo_id), column
            ]
            if np.isnan(value.values[0]):
                return False
            else:
                return True
        except KeyError:
            return False

    def add_outflow_parameters(self):
        if self.snap_range is not None:
            iteration_df = self.df
        else:
            iteration_df = self.df
        counter = 0
        for _, element in iteration_df.iterrows():
            counter += 1
            halo_id = int(element.Halo_id)
            snap = int(element.snap)
            # if self._has_value(halo_id=halo_id, snap=snap):
            #     continue
            if self.with_quantile:
                out_props = self.quantile_outflow_props(halo_id, snap)
            else:
                out_props = self.outflow_props(halo_id, snap)
            for key in out_props.keys():
                self._create_key_column(key)
                self.df.loc[
                    (self.df.snap == snap) & (self.df.Halo_id == halo_id), key
                ] = out_props[key]

            if counter % 100 == 0:
                print(f"Processed {counter/len(iteration_df)*100:.2f}% of galaxies")
                self.save_df()
        return

    def add_outflow_metallicity(self):
        # if self.snap_range is not None:
        #     iteration_df = self.df
        # else:
        iteration_df = self.df
        counter = 0
        if "outflow_Z" not in self.df.keys():
            self.df["outflow_Z"] = np.nan * np.ones(len(self.df))
            self.df["remain_Z"] = np.nan * np.ones(len(self.df))

            # self.df["outflow_Z_warm"] = np.nan * np.ones(len(self.df))

        for _, element in iteration_df.iterrows():
            counter += 1
            halo_id = int(element.Halo_id)
            snap = int(element.snap)
            gal = Galaxy(
                df=self.df,
                halo_id=halo_id,
                snap=snap,
                group_props=self.group_props,
            )
            try:
                self.df.loc[
                    (self.df.snap == snap) & (self.df.Halo_id == halo_id),
                    "outflow_Z",
                ] = gal.get_outflow_metallicity(cold_only=False, type="out")

                self.df.loc[
                    (self.df.snap == snap) & (self.df.Halo_id == halo_id),
                    "remain_Z",
                ] = gal.get_outflow_metallicity(cold_only=False, type="remain")
            except:
                self.df.loc[
                    (self.df.snap == snap) & (self.df.Halo_id == halo_id),
                    "outflow_Z",
                ] = np.nan

                self.df.loc[
                    (self.df.snap == snap) & (self.df.Halo_id == halo_id),
                    "remain_Z",
                ] = np.nan

            if counter % 100 == 0:
                print(f"Processed {counter/len(iteration_df)*100:.2f}% of galaxies")
                self.save_df()
        return

    def get_offset_W80(self, gas):
        try:
            v_range = np.quantile(gas["los_Velocities"], [0.05, 0.1, 0.9, 0.95])
            delta_v = (v_range[3] + v_range[0]) / 2
            W80 = v_range[2] - v_range[1]
        except:
            delta_v = np.nan
            W80 = np.nan
        return delta_v, W80

    def add_outflow_W80(self):
        iteration_df = self.df
        counter = 0
        phi_angles = [0, 30, 60, 90]
        theta_angles = [0, 30, 60, 90, 120, 150]
        keys = ["W80_outflow", "delta_v_outflow", "W80_galaxy", "delta_v_galaxy"]
        for phi in phi_angles:
            for theta in theta_angles:
                for key in keys:
                    if f"{key}_{phi}_{theta}" not in self.df.keys():
                        self.df[f"{key}_{phi}_{theta}"] = np.nan * np.ones(len(self.df))

        for _, element in iteration_df.iterrows():
            counter += 1
            halo_id = int(element.Halo_id)
            snap = int(element.snap)
            for phi in phi_angles:
                for theta in theta_angles:
                    gal = GalaxyProjections(
                        df=self.df,
                        halo_id=halo_id,
                        snap=snap,
                        projection_angle_theta=theta,
                        projection_angle_phi=phi,
                    )
                    try:
                        gal.project_outflows()

                        delta_v_galaxy, W80_galaxy = self.get_offset_W80(gal.remain_gas)
                        delta_v_outflow, W80_outflow = self.get_offset_W80(gal.out_gas)
                    except:
                        delta_v_galaxy = np.nan
                        W80_galaxy = np.nan
                        delta_v_outflow = np.nan
                        W80_outflow = np.nan

                    self.df.loc[
                        (self.df.snap == snap) & (self.df.Halo_id == halo_id),
                        f"delta_v_galaxy_{phi}_{theta}",
                    ] = delta_v_galaxy

                    self.df.loc[
                        (self.df.snap == snap) & (self.df.Halo_id == halo_id),
                        f"W80_galaxy_{phi}_{theta}",
                    ] = W80_galaxy

                    self.df.loc[
                        (self.df.snap == snap) & (self.df.Halo_id == halo_id),
                        f"delta_v_outflow_{phi}_{theta}",
                    ] = delta_v_outflow

                    self.df.loc[
                        (self.df.snap == snap) & (self.df.Halo_id == halo_id),
                        f"W80_outflow_{phi}_{theta}",
                    ] = W80_outflow

            if counter % 100 == 0:
                print(f"Processed {counter/len(iteration_df)*100:.2f}% of galaxies")
                self.save_df()
        return

    def save_df(self):
        self.df.to_hdf(self.save_path, "galaxies")


if __name__ == "__main__":
    updater = OutflowPropUpdater(
        "all_galaxies_extended",
        # "updated_with_W80_angles",
        # "updated_with_M_out",
        # save_name="test",
        save_name="updated_with_W80_angles",
        # save_name="test",
        snap_range=[17, 25],
        # snap_range=[0, 5],
        with_quantile=False,
        only_shell=False,
    )
    # updater.add_outflow_parameters()
    updater.add_outflow_W80()
    # updater.add_outflow_metallicity()

    updater.save_df()
