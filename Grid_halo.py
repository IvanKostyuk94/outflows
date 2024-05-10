import numpy as np
from config import config
from pyTNG import gridding
from process_gas import Galaxy

from scipy.spatial.transform import Rotation as R


class GasGridder:
    def __init__(
        self,
        gal,
        out_only=False,
        quants=None,
        threshold_velocity=None,
        v_esc_ratio=None,
        grid_size=100,
        n_threads=8,
        projection_angle=None,
        grouped_selection=False,
    ):
        self._quants = quants

        self.gal = gal
        self.out_only = out_only
        self.threshold_velocity = threshold_velocity
        self.v_esc_ratio = v_esc_ratio

        self.n_threads = n_threads

        self.grouped_selection = grouped_selection

        self.box_size = (
            0.7 * self.gal.cut_factor * self.gal.scale_radius * 2 * np.ones(3)
        )
        self.shape = (grid_size * np.ones(3)).astype(np.int64)
        self.grid_cen = np.array([0, 0, 0])

    @property
    def quants(self):
        if self._quants is None:
            self._quants = [
                "Temperature",
                "GFM_Metallicity",
                "Flow_Velocities",
                "Rot_Velocities",
                "Angular_Velocities",
            ]
        return self._quants

    def get_gridded(self, gas):
        box_size_parts = [0, 0, 0]
        skip_quants = {"StarFormationRate", "Masses"}
        filtered_quants = [
            quant for quant in self.quants if quant not in skip_quants
        ]
        if len(filtered_quants) == 0:
            filtered_quants = None
        grids = gridding.depositParticlesOnGrid(
            gas_parts=gas,
            method="sphKernelDep",
            quants=filtered_quants,
            box_size_parts=box_size_parts,
            grid_shape=self.shape,
            grid_size=self.box_size,
            grid_cen=self.grid_cen,
            n_threads=self.n_threads,
        )
        if "StarFormationRate" in self.quants:
            grid_sfr = gridding.depositParticlesOnGrid(
                gas_parts=gas,
                method="sphKernelDep",
                quants=[],
                box_size_parts=box_size_parts,
                grid_shape=self.shape,
                grid_size=self.box_size,
                grid_cen=self.grid_cen,
                n_threads=self.n_threads,
                mass_key="StarFormationRate",
            )
            grids["StarFormationRate"] = grid_sfr["StarFormationRate"]
        return grids

    def normalize(self, grids):
        skip_quants = {"StarFormationRate", "Masses"}
        for quant in self.quants:
            if quant not in skip_quants:
                grids[quant] = np.where(
                    grids["Masses"] != 0, grids[quant] / grids["Masses"], 0
                )
        print(np.sum(grids["StarFormationRate"]))
        return

    def grid_gas(self):
        self.gal.rotate_into_galactic_plane()
        if self.out_only:
            self.gal.select_outflowing_gas(
                threshold_velocity=self.threshold_velocity,
                v_esc_ratio=self.v_esc_ratio,
            )

        grids = self.get_gridded(gas=self.gal.gas)
        self.normalize(grids)
        self.grids = grids

        if self.grouped_selection:
            all_grids = []
            all_grids.append(grids)

            galaxy_gridded = self.get_gridded(gas=self.gal.out_galaxy)
            self.normalize(galaxy_gridded)
            all_grids.append(galaxy_gridded)

            outflow_gridded = self.get_gridded(gas=self.gal.out_gas)
            self.normalize(outflow_gridded)
            all_grids.append(outflow_gridded)

            self.grids = all_grids
        return


# Ignore for now to be moved to to new class when developing projection
def line_of_sight_projection(gas, angle, center):
    los_dir = np.array([np.cos(angle), 0, np.sin(angle)])
    los_rot_matrix = [
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)],
    ]
    proj_velocities = np.float32(
        np.multiply(gas["Velocities"], los_dir).sum(axis=1)
    )
    gas["Projected_Velocities"] = proj_velocities
    rot_transformer = R.from_matrix(los_rot_matrix)
    gas["Coordinates"] = rot_transformer.apply(gas["Coordinates"])
    center = rot_transformer.apply(center)
    return center
