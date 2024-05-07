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
        threshold_velocity=None,
        v_esc_ratio=None,
        grid_size=100,
        n_threads=8,
        projection_angle=None,
        grouped_selection=False,
    ):
        self.gal = gal
        self.out_only = out_only
        self.threshold_velocity = threshold_velocity
        self.v_esc_ratio = v_esc_ratio

        self.n_threads = n_threads

        self.grouped_selection = grouped_selection
        self.get_gridding_quants()

        self.box_size = (
            self.gal.cut_factor * self.gal.scale_radius * 2 * np.ones(3)
        )
        self.shape = (grid_size * np.ones(3)).astype(np.int64)
        self.grid_cen = np.array([0, 0, 0])

    def get_gridding_quants(self):
        self.quants = [
            "Temperature",
            "GFM_Metallicity",
            "Flow_Velocities",
        ]
        return

    def get_gridded(self, gas):
        box_size_parts = [0, 0, 0]
        grids = gridding.depositParticlesOnGrid(
            gas_parts=gas,
            method="sphKernelDep",
            quants=self.quants,
            box_size_parts=box_size_parts,
            grid_shape=self.shape,
            grid_size=self.box_size,
            grid_cen=self.grid_cen,
            n_threads=self.n_threads,
        )

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

    def remove_nans(self, grids):
        for quant in self.quants:
            grids[quant] = np.where(
                grids["Masses"] != 0, grids[quant] / grids["Masses"], 0
            )
        return

    def grid_gas(self):
        self.gal.retrieve_halo_gas()
        self.gal.cut_gal_scale()
        self.gal.rotate_into_galactic_plane()
        self.gal.preprocess_for_gridding()

        if self.out_only:
            self.gal.select_outflowing_gas(
                threshold_velocity=self.threshold_velocity,
                v_esc_ratio=self.v_esc_ratio,
            )

        grids = self.get_gridded(gas=self.gal.gas)
        self.remove_nans(grids)
        self.grids = grids

        if self.grouped_selection:
            self.gal.select_outflowing_gas(threshold_velocity=0)
            self.gal.preprocess_for_gridding()

            all_grids = []
            all_grids.append(grids)

            self.gal.group_gas()
            self.gal.get_gas_groups()
            self.gal.select_galaxy_group()
            galaxy_gridded = self.get_gridded(gas=self.gal.galaxy_group)
            all_grids.append(galaxy_gridded)

            self.gal.get_only_outflowing_gas()
            outflow_gridded = self.get_gridded(gas=self.gal.grouped_out_gas)
            all_grids.append(outflow_gridded)

            for grids in all_grids:
                self.remove_nans(grids)
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
