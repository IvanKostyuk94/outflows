import numpy as np
import illustris_python as il
from pyTNG import gas_temperature
from scipy.spatial.transform import Rotation as R
from scipy.integrate import cumtrapz
from gaussian_outflow_selection import (
    group_gas,
    select_galaxy_group,
    get_only_outflowing_gas,
)
from utils import (
    get_sim,
    get_redshift,
    scale_factor,
    get_dm_mass,
    get_dist_in_km,
    map_to_new_dict,
    sort_all_keys,
    calculate_acc,
)


class Galaxy:

    _, sim_path = get_sim()
    out_gas_selections = ["GMM", "v_esc_ratio", "v_crit"]

    def __init__(
        self,
        df,
        halo_id,
        snap,
        cut_factor=10,
        n_peak=4,
        group_props=None,
        out_gas_sel="GMM",
        v_esc_ratio=None,
    ):

        self._halo = None
        self._gal_pos = None
        self._gal_vel = None
        self._stars = None
        self._gas = None
        self._out_gas = None
        self._selected_out_gas = None
        self._out_galaxy = None
        self._ang_mom_dir = None

        self.df = df
        self.halo_id = halo_id
        self.snap = snap
        self.galaxy_id = int(self.halo.idx)
        self.scale_radius = float(self.halo.r_SFR)
        self.cut_factor = cut_factor
        self.r_vir = float(self.halo.R_vir)
        self.n_peak = n_peak
        self.group_props = group_props
        self.v_esc_ratio = v_esc_ratio
        self.z = get_redshift(self.snap)
        if out_gas_sel in self.out_gas_selections:
            self.out_gas_sel = out_gas_sel
        else:
            raise NotImplementedError(
                f"Outflow selection based on {out_gas_sel} is not implemented yet"
            )

    @property
    def halo(self):
        if self._halo is None:
            self._halo = self.df[
                (self.df.Halo_id == self.halo_id) & (self.df.snap == self.snap)
            ]
        return self._halo

    @property
    def gal_pos(self):
        if self._gal_pos is None:
            self._gal_pos = np.array(
                [
                    float(self.halo.Galaxy_pos_x),
                    float(self.halo.Galaxy_pos_y),
                    float(self.halo.Galaxy_pos_z),
                ]
            )
        return self._gal_pos

    @property
    def gal_vel(self):
        if self._gal_vel is None:
            self._gal_vel = np.array(
                [
                    float(self.halo.Galaxy_vel_x),
                    float(self.halo.Galaxy_vel_y),
                    float(self.halo.Galaxy_vel_z),
                ]
            )
        return self._gal_vel

    @property
    def stars(self):
        if self._stars is None:
            self._stars = il.snapshot.loadSubhalo(
                self.sim_path, self.snap, self.galaxy_id, "stars"
            )
            self._get_relative_coordinates(self._stars)
            self._get_relative_distances(self._stars)
            self._get_velocities(self._stars)
        return self._stars

    @property
    def gas(self):
        if self._gas is None:
            self._gas = il.snapshot.loadHalo(
                self.sim_path, self.snap, self.halo_id, "gas"
            )
            if self.out_gas_sel == "v_esc_ratio":
                self._get_gas_v_esc(self, self._gas)

            self._get_relative_coordinates(self._gas)
            self._get_relative_distances(self._gas)
            self._gas = self._cut_gal_scale(self._gas)
            self._get_velocities(self._gas)
            self._get_dir(self._gas)
            self._get_flow(self._gas)
            self._get_rot_vel(self._gas)
            gas_temperature.gasTemp(self._gas)
            self._get_hsml()
        return self._gas

    # Selects all gas that is moving out: v_out>0
    @property
    def out_gas(self):
        if self._out_gas is None:
            if self.out_gas_sel == "GMM":
                all_out_gas = self._select_moving_gas(threshold_velocity=0)
                self._group_gas(all_out_gas)
                galaxy_groups = self._get_gas_groups(all_out_gas)
                galaxy_group_num, self._out_galaxy = self._select_galaxy_group(
                    galaxy_groups
                )
                self._out_gas = get_only_outflowing_gas(
                    all_out_gas, galaxy_group=galaxy_group_num
                )
            elif self.out_gas_sel == "v_esc_ratio":
                self._out_gas = self._select_moving_gas(
                    v_esc_ratio=self.v_esc_ratio
                )
            elif self.out_gas_selections == "v_crit":
                self._out_gas = self._select_moving_gas(
                    threshold_velocity=self.v_esc_ratio
                )
        return self._out_gas

    """Selects the part of the central galaxy that is outflowing.
    I only use it for some tests and it only works for GMM selection
    otherwise it returns None"""

    @property
    def out_galaxy(self):
        if self._out_galaxy is None:
            _ = self.out_gas
        return self._out_galaxy

    @property
    def ang_mom_dir(self):
        if self._ang_mom_dir is None:
            idces_rel_stars = (
                self.stars["Relative_Distances"] < self.scale_radius
            )
            rel_stars = map_to_new_dict(self.stars, idces_rel_stars)
            ang_mom = (
                np.cross(
                    rel_stars["Coordinates"],
                    rel_stars["Relative_Velocities"],
                ).T
                * rel_stars["Masses"]
            )
            tot_ang_mom = np.sum(ang_mom, axis=1)
            self._ang_mom_dir = tot_ang_mom / np.linalg.norm(tot_ang_mom)
        return self._ang_mom_dir

    def _select_moving_gas(self, threshold_velocity=None, v_esc_ratio=None):
        if self.gas["count"] == 0:
            return self.gas
        else:
            if (threshold_velocity is not None) and (v_esc_ratio is None):
                idces_rel_gas = (
                    self.gas["Flow_Velocities"] > threshold_velocity
                )
            elif (threshold_velocity is None) and (v_esc_ratio is not None):
                idces_rel_gas = (
                    self.gas["Relative_Velocities_abs"]
                    > v_esc_ratio * self.gas["v_esc"]
                ) & (self.gas["Flow_Velocities"] > 0)
            else:
                raise ValueError(
                    'Either "v_esc_ratio" or "threshold_velocity" but not both have to be "None"'
                )
            moving_gas = map_to_new_dict(self.gas, idces_rel_gas)
            return moving_gas

    def _cut_gal_scale(self, gas):
        max_r = self.cut_factor * self.scale_radius
        if max_r > self.r_vir:
            max_r = self.r_vir
        relevant_gas = gas["Relative_Distances"] < max_r
        gas = map_to_new_dict(gas, relevant_gas)
        return gas

    def _get_relative_coordinates(self, particles):
        particles["Abs_Coordinates"] = np.copy(particles["Coordinates"])
        particles["Coordinates"] = particles["Coordinates"] - self.gal_pos
        return

    def _get_relative_distances(self, particles):
        particles["Relative_Distances"] = np.sqrt(
            np.sum(np.square(particles["Coordinates"]), axis=1)
        )
        return

    def _get_velocities(self, particles):
        particles["Velocities"] = particles["Velocities"] * np.sqrt(
            scale_factor(self.z)
        )

        particles["Relative_Velocities"] = (
            particles["Velocities"] - self.gal_vel.T
        )
        particles["Relative_Velocities_abs"] = np.linalg.norm(
            particles["Relative_Velocities"], axis=1
        )
        return

    def _get_dir(self, gas):
        gas["Direction"] = (
            gas["Coordinates"].T / np.linalg.norm(gas["Coordinates"], axis=1)
        ).T
        return

    def _get_flow(self, gas):
        # >0 means outflow, <0 means infall
        gas["Flow_Velocities"] = np.float32(
            np.multiply(gas["Relative_Velocities"], gas["Direction"]).sum(
                axis=1
            )
        )
        return

    def _get_rot_vel(self, gas):
        ang_momementa = np.cross(
            gas["Coordinates"],
            gas["Relative_Velocities"],
        ).T
        gas["Rot_Velocities"] = np.float32(
            np.dot(self.ang_mom_dir, ang_momementa)
        )
        gas["Angular_Velocities"] = np.float32(
            (gas["Rot_Velocities"] / gas["Relative_Distances"])
        )
        return

    def _get_hsml(self):
        self.gas["hsml"] = 2.5 * (
            3 * (self.gas["Masses"] / self.gas["Density"]) / (4.0 * np.pi)
        ) ** (1.0 / 3)
        return

    def _group_gas(self, gas):
        group_gas(gas, props=self.group_props, n_peak=self.n_peak)
        return

    def _get_gas_groups(self, gas):
        gas_groups = []
        for i in range(np.max(gas["group"]) + 1):
            gas_group = self.select_gas_group(gas, i)
            gas_groups.append(gas_group)
        return gas_groups

    def _select_galaxy_group(self, gas_groups):
        galaxy_group_num = select_galaxy_group(gas_groups)
        galaxy_group = gas_groups[galaxy_group_num]
        return galaxy_group_num, galaxy_group

    def build_all_particles_dict(self, halo_particles):
        tot_num = np.sum(particles["count"] for particles in halo_particles)
        all_particles = {}
        all_particles["Masses"] = np.empty(tot_num)
        all_particles["Relative_Distances"] = np.empty(tot_num)
        all_particles["Numbering"] = np.empty(tot_num)

        for i, particles in enumerate(halo_particles):
            if i == 0:
                start = 0
            else:
                start = start + halo_particles[i - 1]["count"]
            end = particles["count"] + start
            self._get_relative_coordinates(particles)
            self._get_relative_distances(particles)
            if "Masses" in particles.keys():
                all_particles["Masses"][start:end] = particles["Masses"]
            else:
                dm_mass = get_dm_mass(self.snap)
                all_particles["Masses"][start:end] = dm_mass * np.ones(
                    particles["count"]
                )
            all_particles["Relative_Distances"][start:end] = particles[
                "Relative_Distances"
            ]
            particles["Numbering"] = np.arange(start, end)
            all_particles["Numbering"][start:end] = particles["Numbering"]

        sort_all_keys(particles=all_particles, sort_key="Relative_Distances")
        return all_particles

    def _get_gas_v_esc(self, gas):
        dm = il.snapshot.loadHalo(self.sim_path, self.snap, self.halo_id, "dm")
        stars = il.snapshot.loadHalo(
            self.sim_path, self.snap, self.halo_id, "stars"
        )
        halo_particles = [gas, dm, stars]
        all_particles = self.build_all_particles_dict(self, halo_particles)
        all_particles["Masses_Cum"] = np.cumsum(all_particles["Masses"])

        all_particles["Relative_Distances_km"] = get_dist_in_km(
            all_particles["Relative_Distances"], self.z
        )
        all_particles["Grav_acc"] = calculate_acc(
            all_particles["Masses_Cum"], all_particles["Relative_Distances_km"]
        )
        all_particles["Grav_pot"] = cumtrapz(
            all_particles["Grav_acc"],
            x=all_particles["Relative_Distances_km"],
            initial=0,
        )
        all_particles["v_esc"] = np.sqrt(
            2
            * (
                all_particles["Grav_pot"]
                - all_particles["Grav_acc"][-1]
                * all_particles["Relative_Distances_km"][-1]
                - all_particles["Grav_pot"][-1]
            )
        )

        sort_all_keys(particles=all_particles, sort_key="Numbering")

        gas_indices = np.where(
            np.isin(all_particles["Numbering"], gas["Numbering"])
        )[0]

        gas["v_esc"] = all_particles["v_esc"][gas_indices]
        return gas

    def select_gas_group(self, gas, group_num):
        if gas["count"] == 0:
            return gas
        else:
            idces_rel_gas = gas["group"] == group_num
            rel_gas = map_to_new_dict(gas, idces_rel_gas)
        return rel_gas

    def _rot_matrix_from_ang_mom(self):
        z_axis = np.array([[0, 0, 1]])
        ang_mom_dir_vec = np.array([self.ang_mom_dir])
        rotation, _ = R.align_vectors(z_axis, ang_mom_dir_vec)
        return rotation

    def rotate_into_galactic_plane(self):
        rotation = self._rot_matrix_from_ang_mom()
        self.gas["Coordinates"] = rotation.apply(self.gas["Coordinates"])
        self.gas["Relative_Velocities"] = rotation.apply(
            self.gas["Relative_Velocities"]
        )
        return

    def get_outflow_mass(self):
        out_mass = np.sum(self.out_gas["Masses"])
        return out_mass

    def get_average_outflow_vel(self, weighting="Masses"):
        if weighting == "Luminosity":
            weights = self.out_gas["Density"] * self.out_gas["Masses"]
        else:
            weights = self.out_gas[weighting]
        v_mean = np.average(self.out_gas["Flow_Velocities"], weights=weights)
        return v_mean
