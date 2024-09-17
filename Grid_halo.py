import numpy as np
from pyTNG import gridding

from utils import scale_factor
from pyTNG.cosmology import TNGcosmo
from los_projection import GalaxyProjections


class GasGridder(GalaxyProjections):
    def __init__(
        self,
        df,
        halo_id,
        snap,
        group_props=None,
        out_gas_sel="GMM",
        quants=None,
        grid_size=100,
        n_threads=8,
        projection_angle_theta=None,
        projection_angle_phi=0,
    ):
        super().__init__(
            df,
            halo_id,
            snap,
            group_props=group_props,
            out_gas_sel=out_gas_sel,
            projection_angle_theta=projection_angle_theta,
            projection_angle_phi=projection_angle_phi,
        )
        self._quants = quants
        self._grids = None

        self.n_threads = n_threads

        if self.fixed_selection:
            self.box_size = (
                0.7 * self.cut_r * (1+self.z) * 2 * np.ones(3)
            )
        else:
            self.box_size = (
                0.7 * self.cut_r *  2 * np.ones(3)
            )
        self.shape = (grid_size * np.ones(3)).astype(np.int64)
        self.grid_cen = np.array([0, 0, 0])

        if projection_angle_theta is not None:
            # self.line_of_sight_projection()
            self.project_outflows()

    @property
    def quants(self):
        if self._quants is None:
            self._quants = [
                "Temperature",
                "GFM_Metallicity",
                "Flow_Velocities",
                "Rot_Velocities",
                "Angular_Velocities",
                "Relative_Velocities_abs",
            ]
        return self._quants

    @property
    def grids(self):
        if self._grids is None:
            grid = self._get_gridded(gas=self.gas)
            self._normalize(grid)
            self._grids = [grid]

            if self.out_gas_sel == "GMM":
                # galaxy_gridded = self._get_gridded(gas=self.out_galaxy)
                # self._normalize(galaxy_gridded)
                # self._grids.append(galaxy_gridded)

                outflow_gridded = self._get_gridded(gas=self.out_gas)
                self._normalize(outflow_gridded)
                self._grids.append(outflow_gridded)

                remain_gridded = self._get_gridded(gas=self.remain_gas)
                self._normalize(remain_gridded)
                self._grids.append(remain_gridded)

            elif (self.out_gas_sel == "v_esc_ratio") or (
                self.out_gas_sel == "v_crit"
            ):
                outflow_gridded = self._get_gridded(gas=self.out_gas)
                self._normalize(outflow_gridded)
                self._grids.append(outflow_gridded)

                remain_gridded = self._get_gridded(gas=self.remain_gas)
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
        if "v_z" in self.quants:
            gas["v_z"] = np.float32(gas["Relative_Velocities"][:, 2])
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
        a = scale_factor(self.z)
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
        cell_size = self.get_pixel_length_abs()
        tot_mass_ax = masses.sum(axis=dir) * 1e10 / TNGcosmo.h
        surface_dens = np.log10(tot_mass_ax / cell_size**2 + 1e-9)
        image[surface_dens < 6.5] = 0
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
            "los_Velocities",
            "Relative_Velocities_abs",
            "v_z",
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
