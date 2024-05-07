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
    def __init__(
        self,
        df,
        halo_id,
        snap,
        with_vesc=False,
        cut_factor=50,
        n_peak=4,
        group_props=None,
    ):
        self.df = df
        self.halo_id = halo_id
        self.snap = snap
        self.halo = self.get_halo()
        self.galaxy_id = int(self.halo.idx)
        self.with_vesc = with_vesc
        _, self.sim_path = get_sim()
        self.get_galaxy_pos()
        self.get_galaxy_vel()
        self.scale_radius = float(self.halo.Galaxy_SHMR)
        self.cut_factor = cut_factor
        self.r_vir = float(self.halo.R_vir)
        self.n_peak = n_peak
        self.group_props = group_props

    def get_halo(self):
        halo = self.df[
            (self.df.Halo_id == self.halo_id) & (self.df.snap == self.snap)
        ]
        return halo

    def get_galaxy_pos(self):
        self.gal_pos = np.array(
            [
                float(self.halo.Galaxy_pos_x),
                float(self.halo.Galaxy_pos_y),
                float(self.halo.Galaxy_pos_z),
            ]
        )
        return

    def get_galaxy_vel(self):
        self.galaxy_vel = np.array(
            [
                float(self.halo.Galaxy_vel_x),
                float(self.halo.Galaxy_vel_y),
                float(self.halo.Galaxy_vel_z),
            ]
        )
        return

    def get_relative_coordinates(self, particles):
        particles["Relative_Coordinates"] = (
            particles["Coordinates"] - self.gal_pos
        )
        return

    def get_relative_distances(self, particles):
        particles["Relative_Distances"] = np.sqrt(
            np.sum(np.square(particles["Relative_Coordinates"]), axis=1)
        )
        return

    def retrieve_galaxy_stars(self):
        self.stars = il.snapshot.loadSubhalo(
            self.sim_path, self.snap, self.galaxy_id, "stars"
        )
        self.get_relative_coordinates(self.stars)
        self.get_relative_distances(self.stars)
        self.get_velocities(self.stars)
        return

    def get_velocities(self, particles):
        self.z = get_redshift(self.snap)
        particles["Velocities"] = particles["Velocities"] * np.sqrt(
            scale_factor(self.z)
        )

        self.get_galaxy_vel()
        particles["Relative_Velocities"] = (
            particles["Velocities"] - self.galaxy_vel.T
        )
        particles["Relative_Velocities_abs"] = np.linalg.norm(
            particles["Relative_Velocities"], axis=1
        )
        return

    def get_rel_coord_dist(self):
        self.get_relative_coordinates(self.gas)
        self.gas["Direction"] = (
            self.gas["Relative_Coordinates"].T
            / np.linalg.norm(self.gas["Relative_Coordinates"], axis=1)
        ).T

    def get_flow(self):
        # >0 means outflow, <0 means infall
        self.gas["Flow_Velocities"] = np.float32(
            np.multiply(
                self.gas["Relative_Velocities"], self.gas["Direction"]
            ).sum(axis=1)
        )

    def get_rot_vel(self):
        self.gas["Rot_Velocities"] = np.sqrt(
            self.gas["Relative_Velocities_abs"] ** 2
            - self.gas["Flow_Velocities"] ** 2
        )
        self.gas["Rot_Velocities"][np.isnan(self.gas["Rot_Velocities"])] = 0

    def get_hsml(self):
        self.gas["hsml"] = 2.5 * (
            3 * (self.gas["Masses"] / self.gas["Density"]) / (4.0 * np.pi)
        ) ** (1.0 / 3)

    def retrieve_halo_gas(self):
        self.gas = il.snapshot.loadHalo(
            self.sim_path, self.snap, self.halo_id, "gas"
        )
        if self.with_vesc:
            self.get_gas_v_esc(self)

        self.get_rel_coord_dist()
        self.get_velocities(self.gas)
        self.get_flow()
        self.get_rot_vel()
        gas_temperature.gasTemp(self.gas)
        self.get_relative_distances(self.gas)
        self.get_hsml()
        return

    def group_gas(self):
        group_gas(self.out_gas, props=self.group_props, n_peak=self.n_peak)
        return

    def get_gas_groups(self):
        self.gas_groups = []
        for i in range(np.max(self.out_gas["group"]) + 1):
            gas_group = self.select_gas_group(i)
            self.gas_groups.append(gas_group)
        return

    def select_galaxy_group(self):
        self.galaxy_group_num = select_galaxy_group(self.gas_groups)
        self.galaxy_group = self.gas_groups[self.galaxy_group_num]
        return

    def get_only_outflowing_gas(self):
        if not hasattr(self, "self_galaxy_group"):
            self.select_galaxy_group()
        self.grouped_out_gas = get_only_outflowing_gas(
            out_gas=self.out_gas, galaxy_group=self.galaxy_group_num
        )
        return

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
            self.get_relative_coordinates(particles)
            self.get_relative_distances(particles)
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

    def get_gas_v_esc(self):
        dm = il.snapshot.loadHalo(self.sim_path, self.snap, self.halo_id, "dm")
        stars = il.snapshot.loadHalo(
            self.sim_path, self.snap, self.halo_id, "stars"
        )
        halo_particles = [self.gas, dm, stars]
        all_particles = self.build_all_particles_dict(self, halo_particles)
        all_particles["Masses_Cum"] = np.cumsum(all_particles["Masses"])

        z = get_redshift(self.snap)
        all_particles["Relative_Distances_km"] = get_dist_in_km(
            all_particles["Relative_Distances"], z
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
            np.isin(all_particles["Numbering"], self.gas["Numbering"])
        )[0]

        self.gas["v_esc"] = all_particles["v_esc"][gas_indices]
        return self.gas

    def select_outflowing_gas(self, threshold_velocity=None, v_esc_ratio=None):
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
            self.out_gas = map_to_new_dict(self.gas, idces_rel_gas)
        return

    def select_gas_group(self, group_num):
        if self.out_gas["count"] == 0:
            return self.gas
        else:
            idces_rel_gas = self.out_gas["group"] == group_num
            rel_gas = map_to_new_dict(self.out_gas, idces_rel_gas)
        return rel_gas

    def get_angular_momentum_direction(self):
        self.retrieve_galaxy_stars()
        idces_rel_stars = (
            self.stars["Relative_Distances"] < 5 * self.scale_radius
        )
        rel_stars = map_to_new_dict(self.stars, idces_rel_stars)
        ang_mom = (
            np.cross(
                rel_stars["Relative_Coordinates"],
                rel_stars["Relative_Velocities"],
            ).T
            * rel_stars["Masses"]
        )
        tot_ang_mom = np.sum(ang_mom, axis=1)
        self.ang_mom_dir = tot_ang_mom / np.linalg.norm(tot_ang_mom)
        return

    def rot_matrix_from_ang_mom(self):
        z_axis = np.array([[0, 0, 1]])
        self.ang_mom_dir = np.array([self.ang_mom_dir])
        self.rotation, _ = R.align_vectors(self.ang_mom_dir, z_axis)
        return

    def rotate_into_galactic_plane(self):
        self.get_angular_momentum_direction()
        self.rot_matrix_from_ang_mom()
        self.gas["Relative_Coordinates"] = self.rotation.apply(
            self.gas["Relative_Coordinates"]
        )
        self.gas["Relative_Velocities"] = self.rotation.apply(
            self.gas["Relative_Velocities"]
        )
        return

    def cut_gal_scale(self):
        max_r = self.cut_factor * self.scale_radius
        if max_r > self.r_vir:
            max_r = self.r_vir
        relevant_gas = self.gas["Relative_Distances"] < max_r
        self.gas = map_to_new_dict(self.gas, relevant_gas)
        return

    # This is very hacky and should be adjusted in the future
    def preprocess_for_gridding(self):
        if "Abs_Coordinates" not in self.gas.keys():
            self.gas["Abs_Coordinates"] = np.copy(self.gas["Coordinates"])
        self.gas["Coordinates"]
        self.gas["Coordinates"] = self.gas["Relative_Coordinates"]
        if hasattr(self, "out_gas"):
            self.out_gas["Coordinates"] = self.out_gas["Relative_Coordinates"]
            if "Abs_Coordinates" not in self.out_gas.keys():
                self.out_gas["Abs_Coordinates"] = np.copy(
                    self.out_gas["Coordinates"]
                )
