import numpy as np
from process_gas import Galaxy

"""Project the galactic gas as well as the outflow gas along the
line of sight of the observer. After the initial rotation during the 
initialization of the galaxy the galaxy is facing into the 
(0,0,1) (assuming it has a disc like structure). An angle of 0 corresponds
to viewing the galaxy face on and an angle of 90 corresponds to viewing
the galaxy edge on.
"""


class GalaxyProjections(Galaxy):
    def __init__(self, df, halo_id, snap, projection_angle):
        super().__init__(df, halo_id, snap, with_rotation=True)
        self.angle = projection_angle

        self._projected_outflows = None
        self._view_dir = None

    @property
    def view_dir(self):
        if self._view_dir is None:
            rad_angle = self.angle * np.pi / 180
            self._view_dir = np.array(
                [np.sin(rad_angle), 0, np.cos(rad_angle)]
            )
        return self._view_dir

    def project_outflows(self):
        self.out_gas["los_Velocities"] = np.dot(
            self.out_gas["Relative_Velocities"], self.view_dir
        )
        self.gas["los_Velocities"] = np.dot(
            self.gas["Relative_Velocities"], self.view_dir
        )
        self.remain_gas["los_Velocities"] = np.dot(
            self.remain_gas["Relative_Velocities"], self.view_dir
        )
        return
