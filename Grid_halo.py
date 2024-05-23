import numpy as np
from pyTNG import gridding

from utils import get_redshift, scale_factor
from pyTNG.cosmology import TNGcosmo


class GasGridder:
    def __init__(
        self,
        gal,
        quants=None,
        grid_size=100,
        n_threads=8,
        projection_angle=None,
    ):
        self._quants = quants
        self._grids = None

        self.gal = gal

        self.n_threads = n_threads

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

    @property
    def grids(self):
        if self._grids is None:
            grid = self._get_gridded(gas=self.gal.gas)
            self._normalize(grid)
            self._grids = [grid]

            if self.gal.out_gas_sel == "GMM":
                galaxy_gridded = self._get_gridded(gas=self.gal.out_galaxy)
                self._normalize(galaxy_gridded)
                self._grids.append(galaxy_gridded)

                outflow_gridded = self._get_gridded(gas=self.gal.out_gas)
                self._normalize(outflow_gridded)
                self._grids.append(outflow_gridded)

                remain_gridded = self._get_gridded(gas=self.gal.remain_gas)
                print(self.gal.remain_gas["count"])
                self._normalize(remain_gridded)
                self._grids.append(remain_gridded)

            elif (self.gal.out_gas_sel == "v_esc_ratio") or (
                self.gal.out_gas_sel == "v_crit"
            ):
                outflow_gridded = self._get_gridded(gas=self.gal.out_gas)
                self._normalize(outflow_gridded)
                self._grids.append(outflow_gridded)

                remain_gridded = self._get_gridded(gas=self.gal.remain_gas)
                self._normalize(remain_gridded)
                self._grids.append(remain_gridded)

        return self._grids

    def _get_gridded(self, gas):
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

    def _normalize(self, grids):
        skip_quants = {"StarFormationRate", "Masses"}
        for quant in self.quants:
            if quant not in skip_quants:
                grids[quant] = np.where(
                    grids["Masses"] != 0, grids[quant] / grids["Masses"], 0
                )
        return

    def get_pixel_length_abs(self):
        a = scale_factor(self.gal.z)
        pixel_length_com = self.box_size[0] / self.shape[0]
        pixel_length_abs = pixel_length_com / TNGcosmo.h * a
        return pixel_length_abs

    def _get_surface_densities(self, number, dir):
        gas = self.grids[number]["Masses"]
        cell_size = self.get_pixel_length_abs()
        tot_mass_ax = gas.sum(axis=dir) * 1e10 / TNGcosmo.h
        surface_dens = np.log10(tot_mass_ax / cell_size**2 + 1e-9)
        return surface_dens

    def _get_sfr_densities(self, number, dir):
        sfrs = self.grids[number]["StarFormationRate"]
        cell_size = self.get_pixel_length_abs()
        tot_mass_ax = sfrs.sum(axis=dir)
        surface_dens = np.log10(tot_mass_ax / cell_size**2 + 1e-9)
        return surface_dens

    def _get_mass_weighted_image(self, number, dir, prop):
        data = self.grids[number][prop] * self.grids[number]["Masses"]
        masses = self.grids[number]["Masses"]
        image = np.where(
            data.sum(dir) != 0,
            np.true_divide(data.sum(dir), masses.sum(dir)),
            0,
        )
        log_props = {"GFM_Metallicity", "Temperature"}
        if prop in log_props:
            image = np.log10(image + 1e-20)
        return image

    def get_prop_image(
        self,
        number,
        prop,
        dir,
    ):
        mass_weighted_props = {
            "Flow_Velocities",
            "Rot_Velocities",
            "Angular_Velocities",
            "GFM_Metallicity",
            "Temperature",
        }
        if prop in mass_weighted_props:
            image = self._get_mass_weighted_image(number, dir, prop)
        elif prop == "Masses":
            image = self._get_surface_densities(number, dir)
        elif prop == "StarFormationRate":
            image = self._get_sfr_densities(number, dir)
        else:
            raise NotImplementedError(
                f"The property {prop} is not implemented yet"
            )
        return image


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
