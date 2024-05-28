from process_gas import Galaxy
from pyTNG.cosmology import TNGcosmo
from utils import map_to_new_dict
import numpy as np


class GalaxyShells(Galaxy):
    def __init__(self, df, halo_id, snap, radius, thickness=5):
        super().__init__(df, halo_id, snap)
        self.radius = radius
        self.thickness = thickness

        self._shell_gas = None
        self._shell_out_gas = None
        self._cold_shell_out_gas = None

    @property
    def shell_gas(self):
        if self._shell_gas is None:
            relevant_idces = self.gas["Flow_Velocities"] > 0
            relevant_gas = map_to_new_dict(self.gas, relevant_idces)
            self._shell_gas = self._get_gas_in_shell(relevant_gas)
        return self._shell_gas

    @property
    def shell_out_gas(self):
        if self._shell_out_gas is None:
            self._shell_out_gas = self._get_gas_in_shell(self.out_gas)
        return self._shell_out_gas

    @property
    def cold_shell_out_gas(self):
        if self._cold_shell_out_gas is None:
            self._cold_shell_out_gas = self._get_gas_in_shell(
                self.cold_out_gas
            )
        return self._cold_shell_out_gas

    def _get_gas_in_shell(self, gas):
        physical_distances = (
            gas["Relative_Distances"] / (1 + self.z) / TNGcosmo.h
        )
        shell_idces = (
            physical_distances < (self.radius + self.thickness / 2)
        ) & (physical_distances > (self.radius - self.thickness / 2))
        shell_gas = map_to_new_dict(gas, shell_idces)
        return shell_gas

    def get_shell_outflow_vel(
        self, weighting="Masses", cold_only=False, all=False
    ):
        if cold_only:
            gas = self.cold_shell_out_gas
        else:
            gas = self.shell_out_gas
        if all:
            gas = self.shell_gas
        if weighting == "Luminosity":
            weights = gas["Density"] * gas["Masses"]
        else:
            weights = gas[weighting]
        try:
            print(gas["count"])
            v_mean = np.quantile(gas["Flow_Velocities"], q=0.9)
            # v_mean = np.average(gas["Flow_Velocities"], weights=weights)
        except ZeroDivisionError:
            v_mean = None
        return v_mean
