"""Project the galactic gas as well as the outflow gas along the
line of sight of the observer. After the initial rotation during the
initialization of the galaxy the galaxy is facing into the
(0,0,1) (assuming it has a disc like structure). An angle of 0 corresponds
to viewing the galaxy face on and an angle of 90 corresponds to viewing
the galaxy edge on.
"""
import numpy as np
from process_gas import Galaxy
from utils import map_to_new_dict


class GalaxyProjections(Galaxy):
    def __init__(
        self,
        df,
        halo_id,
        snap,
        projection_angle_theta,
        aperture_size=0.6,
        projection_angle_phi=0,
        group_props=None,
        out_gas_sel="GMM",
        backend=None,
    ):
        super().__init__(
            df,
            halo_id,
            snap,
            with_rotation=True,
            group_props=group_props,
            out_gas_sel=out_gas_sel,
            aperture_size=aperture_size,
            backend=backend,
        )
        self.angle_phi = projection_angle_phi
        self.angle_theta = projection_angle_theta

        self._projected_outflows = None
        self._view_dir = None

    @property
    def view_dir(self):
        if self._view_dir is None:
            rad_angle_theta = self.angle_theta * np.pi / 180
            rad_angle_phi = self.angle_phi * np.pi / 180

            self._view_dir = np.array(
                [
                    np.cos(rad_angle_phi) * np.sin(rad_angle_theta),
                    np.sin(rad_angle_phi) * np.sin(rad_angle_theta),
                    np.cos(rad_angle_theta),
                ]
            )
        return self._view_dir

    def project_outflows(self):
        self.out_gas["los_Velocities"] = np.float32(
            np.dot(self.out_gas["Relative_Velocities"], self.view_dir)
        )
        self.gas["los_Velocities"] = np.float32(
            np.dot(self.gas["Relative_Velocities"], self.view_dir)
        )
        self.remain_gas["los_Velocities"] = np.float32(
            np.dot(self.remain_gas["Relative_Velocities"], self.view_dir)
        )
        self.out_galaxy["los_Velocities"] = np.float32(
            np.dot(self.out_galaxy["Relative_Velocities"], self.view_dir)
        )
        return

    def select_warm_gas(self, gas):
        idces = (gas["Temperature"] > 1e4) & (gas["Temperature"] < 1e5)
        relevant_gas = map_to_new_dict(gas, idces)
        return relevant_gas

    def use_only_warm(self):
        self.out_gas_warm = self.select_warm_gas(self.out_gas)
        self.gas_warm = self.select_warm_gas(self.gas)
        self.remain_gas_warm = self.select_warm_gas(self.remain_gas)
        return
