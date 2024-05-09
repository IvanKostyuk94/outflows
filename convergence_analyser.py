import numpy as np
import pandas as pd
from process_gas import Galaxy


class ConvergenceAnalyser:
    valid_sampling = {"Masses"}
    m_sun_conversion = 1e10 / 0.6774

    def __init__(
        self,
        df,
        sample_size=5,
        sample_by="Masses",
    ):
        self.df = df
        self.sample_size = sample_size

        if sample_by in self.valid_sampling:
            self.samle_by = sample_by
        else:
            raise NotImplementedError(
                f"Sampling by {sample_by} is not implemented yet"
            )
        self._sample = None

    @property
    def sample(self):
        if self._sample is None:
            bins = self._get_bins()
            sample_indices = (
                self.df.groupby(bins)
                .apply(lambda x: x.sample(1))
                .index.droplevel(0)
            )
            self._sample = self.df.loc[sample_indices]
        return self._sample

    def _get_bins(self):
        if self.samle_by == "Masses":
            self.df["log_M_star"] = np.log10(
                self.df["Galaxy_M_star"] * self.m_sun_conversion
            )
            bins = pd.cut(
                self.df["log_M_star"], bins=self.sample_size, labels=False
            )
            return bins

    def get_outflow_props(self, convergence_prop="n_peak"):
        if convergence_prop == "n_peak":
            outflow_dict = self._n_peak_convergence()
        elif convergence_prop == "group_props":
            outflow_dict = self._prop_convergence()
        else:
            raise NotImplementedError(
                f"Testing convergence in {convergence_prop} is not implemented yet"
            )
        return outflow_dict

    def _n_peak_convergence(self):
        peaks = np.arange(2, 15)
        out_mass = {}
        out_vel_lum = {}
        out_vel_mass = {}
        star_masses = []
        for _, element in self.sample.iterrows():
            group_name = f"{element.log_M_star:.2f}"
            star_masses.append(group_name)
            out_mass[group_name] = []
            out_vel_lum[group_name] = []
            out_vel_mass[group_name] = []
            for peak in peaks:
                halo_id = int(element.Halo_id)
                snap = int(element.snap)
                gal = Galaxy(
                    df=self.df, halo_id=halo_id, snap=snap, n_peak=peak
                )
                out_mass[group_name].append(gal.get_outflow_mass())
                out_vel_lum[group_name].append(
                    gal.get_average_outflow_vel(weighting="Luminosity")
                )
                out_vel_mass[group_name].append(
                    gal.get_average_outflow_vel(weighting="Masses")
                )
        full_dict = {}
        full_dict["peaks"] = peaks
        full_dict["star_masses"] = star_masses
        full_dict["out_mass"] = out_mass
        full_dict["out_vel_lum"] = out_vel_lum
        full_dict["out_vel_mass"] = out_vel_mass
        return full_dict

    def _props_to_test(self):
        props = {}
        props["C_FV"] = ["Coordinates", "Flow_Velocities"]
        props["C_T"] = ["Coordinates", "Temperature"]
        props["C_FV_RV"] = ["Coordinates", "Flow_Velocities", "Rot_Velocities"]
        props["C_SFR"] = ["Coordinates", "StarFormationRate"]
        props["SFR_FV_RV_T_RD"] = [
            "Flow_Velocities",
            "Rot_Velocities",
            "StarFormationRate",
            "Relative_Distances",
            "Temperature",
        ]
        props["All"] = [
            "Coordinates",
            "Flow_Velocities",
            "Rot_Velocities",
            "StarFormationRate",
            "Relative_Distances",
            "Temperature",
        ]
        return props

    def _prop_convergence(self):
        props = self._props_to_test()
        out_mass = {}
        out_vel_lum = {}
        out_vel_mass = {}
        star_masses = []
        for _, element in self.sample.iterrows():
            group_name = f"{element.log_M_star:.2f}"
            star_masses.append(group_name)
            out_mass[group_name] = []
            out_vel_lum[group_name] = []
            out_vel_mass[group_name] = []
            for prop in props.keys():
                halo_id = int(element.Halo_id)
                snap = int(element.snap)
                gal = Galaxy(
                    df=self.df,
                    halo_id=halo_id,
                    snap=snap,
                    group_props=props[prop],
                )
                out_mass[group_name].append(gal.get_outflow_mass())
                out_vel_lum[group_name].append(
                    gal.get_average_outflow_vel(weighting="Luminosity")
                )
                out_vel_mass[group_name].append(
                    gal.get_average_outflow_vel(weighting="Masses")
                )
        full_dict = {}
        full_dict["props"] = props.keys()
        full_dict["star_masses"] = star_masses
        full_dict["out_mass"] = out_mass
        full_dict["out_vel_lum"] = out_vel_lum
        full_dict["out_vel_mass"] = out_vel_mass
        return full_dict
