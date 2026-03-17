import os
import numpy as np
import pandas as pd
from config import config
from process_gas import Galaxy
from galaxy_shell_outflows import GalaxyShells
from los_projection import GalaxyProjections
from tng_cosmo import TNGcosmo


class OutflowPropUpdater:
    def __init__(
        self,
        df_name,
        backend,
        save_name=None,
        group_props=None,
        snap_range=None,
        with_quantile=False,
        only_shell=False,
        in_aperture=False,
        aperture_size=0.3,
    ):
        self.df_name = df_name + config["hdf_ending"]
        self.snap_range = snap_range
        self.save_name = save_name
        self.shell = only_shell
        self.backend = backend

        self.group_props = group_props
        self.with_quantile = with_quantile
        self.in_aperture = in_aperture
        self.aperture_size = aperture_size

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
            backend=self.backend,
            aperture_size=self.aperture_size,
        )
        keys = ["M_out_0.6"]
        out_props = {}
        try:
            out_props["M_out_0.6"] = gal.get_outflow_mass(in_aperture=self.in_aperture)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("outflow_props failed: %s", e)
            for key in keys:
                out_props[key] = None
        return out_props

    def quantile_outflow_props(self, halo_id, snap):
        if self.shell:
            gal = GalaxyShells(
                df=self.df,
                halo_id=halo_id,
                snap=snap,
                radius=10,
                backend=self.backend,
                aperture_size=self.aperture_size,
            )
        else:
            gal = Galaxy(
                df=self.df,
                halo_id=halo_id,
                snap=snap,
                group_props=self.group_props,
                backend=self.backend,
                aperture_size=self.aperture_size,
            )
        quantiles = [0.5, 0.8, 0.9]
        weightings = ["Masses"]
        subscripts = ["mass"]
        keys = [
            f"v_{sub}_{int(q * 100)}"
            for sub, _ in zip(subscripts, weightings)
            for q in quantiles
        ]
        out_vels = {}
        try:
            for sub, weighting in zip(subscripts, weightings):
                for quantile in quantiles:
                    column_name = f"v_{sub}_{int(quantile*100)}"
                    out_vels[column_name] = gal.get_quantile_velocity(
                        quantile, weighting=weighting, in_aperture=self.in_aperture
                    )
        except ValueError:
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
        import logging
        logger = logging.getLogger(__name__)
        iteration_df = self.df
        id_col = self.backend.get_halo_id_column()
        counter = 0
        for _, element in iteration_df.iterrows():
            counter += 1
            halo_id = int(element[id_col])
            snap = int(element.snap)
            if self.with_quantile:
                out_props = self.quantile_outflow_props(halo_id, snap)
            else:
                out_props = self.outflow_props(halo_id, snap)
            for key in out_props.keys():
                self._create_key_column(key)
                self.df.loc[
                    (self.df.snap == snap) & (self.df[id_col] == halo_id), key
                ] = out_props[key]

            if counter % 100 == 0:
                logger.info("Processed %.2f%% of galaxies", counter / len(iteration_df) * 100)
                self.save_df()
        return

    def add_outflow_metallicity(self):
        iteration_df = self.df
        counter = 0
        if "outflow_Z" not in self.df.keys():
            self.df["outflow_Z"] = np.nan * np.ones(len(self.df))
            self.df["remain_Z"] = np.nan * np.ones(len(self.df))

        if self.in_aperture:
            if "outflow_Z_aperture" not in self.df.keys():
                self.df["outflow_Z_aperture"] = np.nan * np.ones(len(self.df))
                self.df["remain_Z_aperture"] = np.nan * np.ones(len(self.df))

            # self.df["outflow_Z_warm"] = np.nan * np.ones(len(self.df))
        id_col = self.backend.get_halo_id_column()
        for _, element in iteration_df.iterrows():
            counter += 1
            halo_id = int(element[id_col])
            snap = int(element.snap)
            gal = Galaxy(
                df=self.df,
                halo_id=halo_id,
                snap=snap,
                group_props=self.group_props,
                aperture_size=self.aperture_size,
                backend=self.backend,
            )
            try:
                loc = (self.df.snap == snap) & (self.df[id_col] == halo_id)
                self.df.loc[loc, "outflow_Z"] = gal.get_outflow_metallicity(
                    cold_only=False, type="out"
                )
                self.df.loc[loc, "remain_Z"] = gal.get_outflow_metallicity(
                    cold_only=False, type="remain"
                )
                if self.in_aperture:
                    try:
                        self.df.loc[loc, "outflow_Z_aperture"] = (
                            gal.get_outflow_metallicity(
                                cold_only=False, type="out", in_aperture=True
                            )
                        )
                        self.df.loc[loc, "remain_Z_aperture"] = (
                            gal.get_outflow_metallicity(
                                cold_only=False, type="remain", in_aperture=True
                            )
                        )
                    except Exception as e:
                        import logging
                        logging.getLogger(__name__).warning("aperture metallicity failed: %s", e)
                        self.df.loc[loc, "outflow_Z_aperture"] = np.nan
                        self.df.loc[loc, "remain_Z_aperture"] = np.nan
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning("metallicity failed: %s", e)
                loc = (self.df.snap == snap) & (self.df[id_col] == halo_id)
                self.df.loc[loc, "outflow_Z"] = np.nan
                self.df.loc[loc, "remain_Z"] = np.nan

            if counter % 100 == 0:
                import logging
                logging.getLogger(__name__).info(
                    "Processed %.2f%% of galaxies",
                    counter / len(iteration_df) * 100,
                )
                self.save_df()
        return

    def get_offset_W80(self, gas):
        try:
            v_range = np.quantile(gas["los_Velocities"], [0.05, 0.1, 0.9, 0.95])
            delta_v = (v_range[3] + v_range[0]) / 2
            W80 = v_range[2] - v_range[1]
        except Exception:
            delta_v = np.nan
            W80 = np.nan
        return delta_v, W80

    def add_outflow_W80(self):
        iteration_df = self.df
        counter = 0
        phi_angles = [0]
        theta_angles = [0, 30, 60, 90]
        keys = [
            "W80_outflow",
            "delta_v_outflow",
            "W80_galaxy",
            "delta_v_galaxy",
            "W80_outflow_aperture",
            "delta_v_outflow_aperture",
            "W80_galaxy_aperture",
            "delta_v_galaxy_aperture",
        ]
        for phi in phi_angles:
            for theta in theta_angles:
                for key in keys:
                    # if f"{key}_{phi}_{theta}" not in self.df.keys():
                    self.df[f"{key}_{phi}_{theta}"] = np.nan * np.ones(len(self.df))
                    if self.in_aperture:
                        self.df[f"{key}_{phi}_{theta}_aperture"] = np.nan * np.ones(
                            len(self.df)
                        )

        id_col = self.backend.get_halo_id_column()
        for _, element in iteration_df.iterrows():
            counter += 1
            halo_id = int(element[id_col])
            snap = int(element.snap)
            loc = (self.df.snap == snap) & (self.df[id_col] == halo_id)
            for phi in phi_angles:
                for theta in theta_angles:
                    gal = GalaxyProjections(
                        df=self.df,
                        halo_id=halo_id,
                        snap=snap,
                        projection_angle_theta=theta,
                        projection_angle_phi=phi,
                        aperture_size=self.aperture_size,
                        backend=self.backend,
                    )
                    try:
                        gal.project_outflows()
                        remain_gas = (
                            gal.get_in_aperture(gal.remain_gas)
                            if self.in_aperture
                            else gal.remain_gas
                        )
                        out_gas = (
                            gal.get_in_aperture(gal.out_gas)
                            if self.in_aperture
                            else gal.out_gas
                        )
                        delta_v_galaxy, W80_galaxy = self.get_offset_W80(remain_gas)
                        delta_v_outflow, W80_outflow = self.get_offset_W80(out_gas)
                    except Exception as e:
                        import logging
                        logging.getLogger(__name__).warning("W80 failed: %s", e)
                        delta_v_galaxy = W80_galaxy = delta_v_outflow = W80_outflow = np.nan

                    self.df.loc[loc, f"delta_v_galaxy_{phi}_{theta}"] = delta_v_galaxy
                    self.df.loc[loc, f"W80_galaxy_{phi}_{theta}"] = W80_galaxy
                    self.df.loc[loc, f"delta_v_outflow_{phi}_{theta}"] = delta_v_outflow
                    self.df.loc[loc, f"W80_outflow_{phi}_{theta}"] = W80_outflow

            if counter % 100 == 0:
                import logging
                logging.getLogger(__name__).info(
                    "Processed %.2f%% of galaxies",
                    counter / len(iteration_df) * 100,
                )
                self.save_df()
        return

    def get_wind_mass(self, halo_id, snap):
        gal = Galaxy(
            df=self.df,
            halo_id=halo_id,
            snap=snap,
            group_props=self.group_props,
            backend=self.backend,
            aperture_size=self.aperture_size,
        )
        try:
            wind_mass = np.sum(gal.wind_aperture["Masses"] * 1e10 / TNGcosmo.h)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("wind mass failed: %s", e)
            wind_mass = None
        return wind_mass

    def add_wind_masses(self):
        if not self.backend.has_wind_particles():
            import logging
            logging.getLogger(__name__).info("Wind mass not available for this backend")
            return
        iteration_df = self.df
        id_col = self.backend.get_halo_id_column()

        wind_col_name = "wind_mass"
        if wind_col_name not in self.df.keys():
            self.df[wind_col_name] = np.nan * np.ones(len(self.df))

        counter = 0
        for _, element in iteration_df.iterrows():
            counter += 1
            halo_id = int(element[id_col])
            snap = int(element.snap)
            wind_mass = self.get_wind_mass(halo_id, snap)
            self.df.loc[
                (self.df.snap == snap) & (self.df[id_col] == halo_id),
                wind_col_name,
            ] = wind_mass
            if counter % 100 == 0:
                import logging
                logging.getLogger(__name__).info(
                    "Processed %.2f%% of galaxies",
                    counter / len(iteration_df) * 100,
                )
                self.save_df()
        return


    def save_df(self):
        self.df.to_hdf(self.save_path, key="galaxies")


if __name__ == "__main__":
    from tng_backend import TNGBackend
    from config import config as _cfg

    backend = TNGBackend(config=_cfg)
    updater = OutflowPropUpdater(
        "in_aperture_final",
        backend=backend,
        save_name="in_aperture_wind",
        snap_range=[13, 26],
        with_quantile=False,
        only_shell=False,
        in_aperture=True,
        aperture_size=0.6,
    )
    updater.add_wind_masses()
    updater.save_df()
