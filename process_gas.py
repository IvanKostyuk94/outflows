import warnings
import numpy as np
import illustris_python as il
from sklearn.decomposition import PCA
import astropy.units as u
import astropy.constants as c
from scipy.spatial.transform import Rotation as R
from scipy.integrate import cumulative_trapezoid as cumtrapz
from astropy.cosmology import Planck18 as cosmo

from utils import (
    get_serra_gas,
    get_serra_stars,
    get_sim,
    get_redshift,
    scale_factor,
    get_dm_mass,
    get_dist_in_km,
    map_to_new_dict,
    sort_all_keys,
    calculate_acc,
)
from pyTNG import gas_temperature
from pyTNG.cosmology import TNGcosmo
from gaussian_outflow_selection import (
    group_gas,
    select_galaxy_group,
    get_only_outflowing_gas,
)


class Galaxy:
    """
    Represents a galaxy (or subhalo) with methods to load and process
    various particle types (gas, stars, etc.) and compute derived quantities.
    """

    out_gas_selections = ["GMM", "v_esc_ratio", "v_crit"]
    _, sim_path = get_sim()

    def __init__(
        self,
        df,
        halo_id,
        snap,
        aperture_size,
        cut_factor=5,
        group_props=None,
        out_gas_sel="GMM",
        v_esc_ratio=0.3,
        with_rotation=False,
        fixed_selection=False,
        serra=False,
    ):
        # Basic configuration
        self.df = df
        self.halo_id = halo_id
        self.snap = snap
        self.serra = serra
        self.fixed_selection = fixed_selection
        self.cut_factor = cut_factor
        self.group_props = group_props
        self.v_esc_ratio = v_esc_ratio
        self.with_rotation = with_rotation
        self.h = 0.6774  # Hubble parameter (for internal conversion)

        # Placeholders for lazy-loaded properties
        self._halo = None
        self._gal_pos = None
        self._gal_vel = None
        self._stars = None
        self._gas = None
        self._wind = None
        self._wind_aperture = None
        self._out_gas = None
        self._out_galaxy = None
        self._cold_out_gas = None
        self._remain_gas = None
        self._ang_mom_dir = None

        # Derived quantities from halo properties
        if not self.serra:
            self.galaxy_id = int(self.halo.idx.iloc[0])
            self.r_vir = float(self.halo.R_vir.iloc[0])
        self.z = self.halo.z.values[0] if self.serra else get_redshift(self.snap)
        self.critical_velocity = 2 * self.halo.SubhaloVelDisp.values[0]
        self.critical_out_velocity = self.critical_velocity

        # Aperture (physical) radius in kpc
        self.aperture_r = cosmo.kpc_proper_per_arcmin(self.z).value * aperture_size / 60

        # Determine the scale radius and number of peaks based on stellar mass log
        if self.serra:
            self.n_peak = 2
        else:
            if self.halo.M_star_log.values[0] < 10.5:
                self.n_peak = 3
            else:
                self.n_peak = 3
        r_sfr = float(self.halo.r_SFR.iloc[0])
        if self.halo.M_star_log.values[0] < 9:
            self.scale_radius = r_sfr
        else:
            self.scale_radius = r_sfr * 2

        # Set the cut radius for particle selection
        self.cut_r = (
            3.5
            if fixed_selection
            else (
                self.aperture_r * 2
                if self.serra
                else self.cut_factor * self.scale_radius
            )
        )

        if not self.serra and self.cut_r > self.r_vir:
            self.cut_r = self.r_vir

        # Validate outflow selection method
        if out_gas_sel in self.out_gas_selections:
            self.out_gas_sel = out_gas_sel
        else:
            raise NotImplementedError(
                f"Outflow selection '{out_gas_sel}' is not implemented yet"
            )

    @property
    def halo(self):
        """Lazy-load the halo dataframe."""
        if self._halo is None:
            if self.serra:
                self._halo = self.df[
                    (self.df.idx == self.halo_id) & (self.df.snap == self.snap)
                ]
            else:
                self._halo = self.df[
                    (self.df.Halo_id == self.halo_id) & (self.df.snap == self.snap)
                ]
        return self._halo

    @property
    def gal_pos(self):
        """Return the galaxy’s position as a numpy array."""
        if self._gal_pos is None:
            # For Serra, this might trigger a side-effect
            if self.serra:
                get_serra_gas(self.snap, self.halo_id)
            else:
                _ = il.snapshot.loadSubhalo(
                    self.sim_path, self.snap, self.galaxy_id, "gas"
                )
            self._gal_pos = np.array(
                [
                    float(self.halo.Galaxy_pos_x.iloc[0]),
                    float(self.halo.Galaxy_pos_y.iloc[0]),
                    float(self.halo.Galaxy_pos_z.iloc[0]),
                ]
            )
        return self._gal_pos

    @property
    def gal_vel(self):
        """Return the galaxy’s velocity as a numpy array."""
        if self._gal_vel is None:
            self._gal_vel = np.array(
                [
                    float(self.halo.Galaxy_vel_x.iloc[0]),
                    float(self.halo.Galaxy_vel_y.iloc[0]),
                    float(self.halo.Galaxy_vel_z.iloc[0]),
                ]
            )
        return self._gal_vel

    @property
    def stars(self):
        """Load and process the star particles."""
        if self._stars is None:
            if self.serra:
                self._stars = get_serra_stars(self.snap, self.halo_id)
            else:
                stars_and_wind = il.snapshot.loadSubhalo(
                    self.sim_path, self.snap, self.galaxy_id, "stars"
                )
                relevant_stars = stars_and_wind["GFM_StellarFormationTime"] > 0
                self._stars = map_to_new_dict(stars_and_wind, relevant_stars)
                wind_test = stars_and_wind["GFM_StellarFormationTime"] < 0
                # self._wind = map_to_new_dict(stars_and_wind, wind_test)
                # wind_mass = np.sum(self._wind['Masses']*1e10/0.6774)
            self._get_relative_coordinates(self._stars)
            self._get_relative_distances(self._stars)
            self._stars = self._cut_gal_scale(self._stars)
            self._get_velocities(self._stars)
        return self._stars
    
    @property
    def wind(self):
        if self._wind is None:
            stars_and_wind = il.snapshot.loadSubhalo(
                        self.sim_path, self.snap, self.galaxy_id, "stars"
                    )
            wind_ids = stars_and_wind["GFM_StellarFormationTime"] <= 0
            self._wind = map_to_new_dict(stars_and_wind, wind_ids)
        return self._wind
    
    @property
    def wind_aperture(self):
        if self._wind_aperture is None:
            wind = self.wind
            self._get_relative_coordinates(wind)
            self._get_relative_distances(wind)
            in_aperture = wind["Physical_Relative_Distances"] < self.aperture_r
            self._wind_aperture = map_to_new_dict(wind, in_aperture)
        return self._wind_aperture
    

    @property
    def gas(self):
        """Load and process the gas particles."""
        if self._gas is None:
            self._gas = (
                get_serra_gas(self.snap, self.halo_id)
                if self.serra
                else il.snapshot.loadHalo(self.sim_path, self.snap, self.halo_id, "gas")
            )
            if self.out_gas_sel == "v_esc_ratio":
                self._get_gas_v_esc(self._gas)
            self._get_relative_coordinates(self._gas)
            self._get_relative_distances(self._gas)
            if not self.serra:
                self._convert_density(self._gas)
            if not self.serra:
                self._gas = self._cut_gal_scale(self._gas)

            if not self.serra:
                self._gas = self._cut_gal_scale(self._gas)
            self._set_idces(self._gas)
            self._get_velocities(self._gas)
            self._get_dir(self._gas)
            self._get_flow(self._gas)
            self._get_rot_vel(self._gas)
            if not self.serra:
                gas_temperature.gasTemp(self._gas)
                self._get_hsml()
            self._gas = self.rotate_into_galactic_plane(self._gas)
            self.get_los_projections()
        return self._gas

    @property
    def out_gas(self):
        """Return the subset of gas defined as outflowing."""
        if self._out_gas is None:
            if self.out_gas_sel == "GMM":
                all_out_gas = self._select_moving_gas(threshold_velocity=0)
                self._group_gas(all_out_gas)
                gas_groups = self._get_gas_groups(all_out_gas)
                galaxy_group_num, self._out_galaxy = self._select_galaxy_group(
                    gas_groups
                )
                outflowing_gas = get_only_outflowing_gas(
                    all_out_gas,
                    galaxy_group=galaxy_group_num,
                    crit_vout=self.critical_out_velocity,
                )
                out_gas_z = self.get_out_gas_z()
                overlap_indices = np.union1d(
                    outflowing_gas["idces"], out_gas_z["idces"]
                )
                result_array = np.full_like(self.gas["idces"], False, dtype=bool)
                result_array[np.isin(self.gas["idces"], overlap_indices)] = True
                self._out_gas = map_to_new_dict(self.gas, result_array)

            elif self.out_gas_sel == "v_esc_ratio":
                self._out_gas = self._select_moving_gas(v_esc_ratio=self.v_esc_ratio)
            elif self.out_gas_sel == "v_crit":
                self._out_gas = self._select_moving_gas(
                    threshold_velocity=self.v_esc_ratio
                )
        return self._out_gas

    @property
    def cold_out_gas(self):
        """Return only the cold outflowing gas (10^4 K < T < 10^5 K)."""
        if self._cold_out_gas is None:
            idces_cold = (1e4 < self.out_gas["Temperature"]) & (
                self.out_gas["Temperature"] < 1e5
            )
            self._cold_out_gas = map_to_new_dict(self._out_gas, idces_cold)
        return self._cold_out_gas

    @property
    def remain_gas(self):
        """Return the gas not identified as outflowing."""
        if self._remain_gas is None:
            overlap_indices = np.setdiff1d(self.gas["idces"], self.out_gas["idces"])
            result_array = np.full_like(self.gas["idces"], False, dtype=bool)
            result_array[np.isin(self.gas["idces"], overlap_indices)] = True
            self._remain_gas = map_to_new_dict(self.gas, result_array)
            # if not self.serra:
            #     not_inflow = (
            #         np.array(self._remain_gas["Relative_Distances"])
            #         < 4 * self.halo.r_SFR.values[0]
            #     )
            #     self._remain_gas = map_to_new_dict(self._remain_gas, not_inflow)
            # else:
            #     not_inflow = (
            #         np.array(self._remain_gas["Relative_Distances"])
            #         < 20 * self.halo.r_SFR.values[0]
            #     )
            #     self._remain_gas = map_to_new_dict(self._remain_gas, not_inflow)
        return self._remain_gas

    @property
    def out_galaxy(self):
        """Return the central outflowing galaxy group (if applicable)."""
        if self._out_galaxy is None:
            _ = self.out_gas  # Triggers outflow selection
        return self._out_galaxy

    @property
    def ang_mom_dir(self):
        """Compute the angular momentum direction of the galaxy (from stars or gas)."""
        if self._ang_mom_dir is None:
            if self.serra:
                idces = self.stars["Relative_Distances"] < 2
                # rel_stars = map_to_new_dict(self.stars, idces)
                # # mass_weighted = rel_stars["Coordinates"] * rel_stars["Masses"][:, np.newaxis]
                # test_vel = rel_stars["Relative_Velocities"]-np.mean(rel_stars["Relative_Velocities"], axis=0)
                # ang_mom = np.cross(
                #     rel_stars["Coordinates"],
                #     rel_stars['Relative_Velocities'],
                # ).T * rel_stars["Masses"]
            else:
                idces = self.stars["Relative_Distances"] < self.scale_radius
            rel_stars = map_to_new_dict(self.stars, idces)
            ang_mom = (
                np.cross(
                    rel_stars["Coordinates"], rel_stars["Relative_Velocities"]
                ).T
                * rel_stars["Masses"]
            )
            tot_ang_mom = np.sum(ang_mom, axis=1)
            self._ang_mom_dir = tot_ang_mom / np.linalg.norm(tot_ang_mom)
        return self._ang_mom_dir

    def _convert_density(self, gas):
        """Convert density to electron density in cgs units."""
        to_msun_ckpc = (1e10 / self.h) / (1 / self.h / (self.halo.z.iloc[0] + 1)) ** 3
        to_cm_3 = to_msun_ckpc * c.M_sun / c.m_p * 0.76 / (u.kpc.to(u.cm)) ** 3
        gas["Density_e"] = gas["Density"] * to_cm_3 * gas["ElectronAbundance"]

    def get_out_gas_z(self):
        """Select gas based on its vertical velocity and coordinate."""
        idces_z = (
            (self.gas["Relative_Velocities"][:, 2] > self.critical_velocity)
            & (self.gas["Coordinates"][:, 2] > 0)
        ) | (
            (self.gas["Relative_Velocities"][:, 2] < -self.critical_velocity)
            & (self.gas["Coordinates"][:, 2] < 0)
        )
        return map_to_new_dict(self.gas, idces_z)

    def _set_idces(self, gas):
        """Set unique indices for each gas particle."""
        gas["idces"] = np.arange(gas["count"])

    def _select_moving_gas(self, threshold_velocity=None, v_esc_ratio=None):
        """Select gas particles based on a threshold velocity or escape velocity ratio."""
        if self.gas["count"] == 0:
            return self.gas
        if threshold_velocity is not None and v_esc_ratio is None:
            idces = self.gas["Flow_Velocities"] > threshold_velocity
        elif threshold_velocity is None and v_esc_ratio is not None:
            idces = (
                self.gas["Relative_Velocities_abs"] > v_esc_ratio * self.gas["v_esc"]
            ) & (self.gas["Flow_Velocities"] > 0)
        else:
            raise ValueError(
                'Either "v_esc_ratio" or "threshold_velocity" must be None.'
            )
        return map_to_new_dict(self.gas, idces)

    def _cut_gal_scale(self, gas):
        """Cut gas based on a fixed or variable radial scale."""
        if self.serra:
            gas['pre_dist'] = np.sqrt(
                np.sum(np.square(gas["Coordinates"]), axis=1)
            ).astype(np.float32)
            selection = gas['pre_dist'] < self.cut_r

        else:
            if self.fixed_selection:
                selection = gas["Physical_Relative_Distances"] < self.cut_r
            else:
                selection = gas["Relative_Distances"] < self.cut_r
        return map_to_new_dict(gas, selection)

    def _get_relative_coordinates(self, particles):
        """Compute absolute and (if applicable) physical coordinates."""
        if "Abs_Coordinates" not in particles:
            particles["Abs_Coordinates"] = np.copy(particles["Coordinates"])
            if not self.serra:
                particles["Coordinates"] -= self.gal_pos
                particles["Physical_Coordinates"] = particles["Coordinates"] / (
                    (1 + self.z) * self.h
                )
            else:
                particles["Physical_Coordinates"] = particles["Coordinates"]

    def _get_relative_distances(self, particles):
        """Compute relative and physical distances from the galaxy center."""
        particles["Relative_Distances"] = np.sqrt(
            np.sum(np.square(particles["Coordinates"]), axis=1)
        ).astype(np.float32)
        particles["Physical_Relative_Distances"] = np.sqrt(
            np.sum(np.square(particles["Physical_Coordinates"]), axis=1)
        )
        if not self.serra:
            particles["SFR_dist"] = particles["Relative_Distances"] / (
                2 * float(self.halo.r_SFR.iloc[0])
            )
            particles["SFR_dist"][particles["SFR_dist"] < 1] = 1

    def _get_velocities(self, particles):
        """Compute velocities relative to the mean velocity."""
        if not self.serra:
            particles["Velocities"] *= np.sqrt(scale_factor(self.z))
        if self.serra:
            # rel_mask = particles["Relative_Distances"] < 20 * self.halo.r_SFR.values[0]
            rel_mask = particles["Relative_Distances"] < 5
            particles_subset = map_to_new_dict(particles, rel_mask)
            mean_vel = np.average(
                particles_subset["Velocities"],
                weights=particles_subset["mass"],
                axis=0,
            )
        else:
            mean_vel = np.average(particles["Velocities"], axis=0)
        particles["Relative_Velocities"] = particles["Velocities"] - mean_vel
        particles["Relative_Velocities_abs"] = np.linalg.norm(
            particles["Relative_Velocities"], axis=1
        ).astype(np.float32)

    def _get_dir(self, gas):
        """Compute the unit vector in the direction of each gas particle."""
        norm = np.linalg.norm(gas["Coordinates"], axis=1)[:, np.newaxis]
        gas["Direction"] = gas["Coordinates"] / norm

    def _get_flow(self, gas):
        """Compute the outflow (or inflow) velocity along the radial direction."""
        gas["Flow_Velocities"] = np.float32(
            (gas["Relative_Velocities"] * gas["Direction"]).sum(axis=1)
        )

    def _get_rot_vel(self, gas):
        """Compute rotational velocities using the angular momentum direction."""
        ang_mom = np.cross(gas["Coordinates"], gas["Relative_Velocities"]).T
        gas["Rot_Velocities"] = np.float32(np.dot(self.ang_mom_dir, ang_mom))
        gas["Angular_Velocities"] = np.float32(
            gas["Rot_Velocities"] / gas["Relative_Distances"]
        )

    def _get_hsml(self):
        """Compute the smoothing length for gas particles."""
        self.gas["hsml"] = 2.5 * (
            3 * (self.gas["Masses"] / self.gas["Density"]) / (4.0 * np.pi)
        ) ** (1.0 / 3)

    def _group_gas(self, gas):
        """Group gas particles using external grouping logic."""
        group_gas(gas, props=self.group_props, n_peak=self.n_peak, serra=self.serra)

    def _get_gas_groups(self, gas):
        """Return a list of gas groups."""
        n_groups = np.max(gas["group"]) + 1
        return [self.select_gas_group(gas, i) for i in range(n_groups)]

    def _select_galaxy_group(self, gas_groups):
        """Select the galaxy group based on external criteria."""
        group_num = select_galaxy_group(gas_groups, serra=self.serra)
        galaxy_group = gas_groups[group_num]
        slow_mask = np.array(galaxy_group["Flow_Velocities"]) < self.critical_velocity
        galaxy_group = map_to_new_dict(galaxy_group, slow_mask)
        return group_num, galaxy_group

    def _build_all_particles_dict(self, halo_particles):
        """Combine particle information from multiple types (gas, DM, stars)."""
        tot_num = sum(p["count"] for p in halo_particles)
        all_particles = {
            "Masses": np.empty(tot_num),
            "Relative_Distances": np.empty(tot_num),
            "Numbering": np.empty(tot_num),
        }
        start = 0
        for particles in halo_particles:
            end = start + particles["count"]
            self._get_relative_coordinates(particles)
            self._get_relative_distances(particles)
            if "Masses" in particles:
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
            start = end
        sort_all_keys(particles=all_particles, sort_key="Relative_Distances")
        return all_particles

    def _get_gas_v_esc(self, gas):
        """Compute the escape velocity for each gas particle."""
        dm = il.snapshot.loadHalo(self.sim_path, self.snap, self.halo_id, "dm")
        stars = il.snapshot.loadHalo(self.sim_path, self.snap, self.halo_id, "stars")
        halo_particles = [gas, dm, stars]
        all_particles = self._build_all_particles_dict(halo_particles)
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
        gas_indices = np.where(np.isin(all_particles["Numbering"], gas["Numbering"]))[0]
        gas["v_esc"] = all_particles["v_esc"][gas_indices]
        return gas

    def select_gas_group(self, gas, group_num):
        """Select a gas group given its group number."""
        if gas["count"] == 0:
            return gas
        idces = gas["group"] == group_num
        return map_to_new_dict(gas, idces)

    def _rot_matrix_from_ang_mom(self):
        """Return a rotation that aligns the galaxy’s angular momentum with the z-axis."""
        z_axis = np.array([[0, 0, 1]])
        ang_mom_vec = np.array([self.ang_mom_dir])
        rotation, _ = R.align_vectors(z_axis, ang_mom_vec)
        return rotation

    def rotate_into_galactic_plane(self, gas):
        """Rotate the gas coordinates into the galactic plane."""
        # if self.serra:
        #     idces = self.gas["Relative_Distances"] < 0.5
        #     rel_gas = map_to_new_dict(self.gas, idces)
        #     if rel_gas["count"] >= 5:
        #         pca = PCA(n_components=3)
        #         pca.fit(rel_gas["Coordinates"] * rel_gas["Masses"][:, np.newaxis])
        #         rotation_matrix = pca.components_
        #         gas["Coordinates"] = gas["Coordinates"].dot(rotation_matrix.T)
        #         gas["Relative_Velocities"] = gas["Relative_Velocities"].dot(
        #             rotation_matrix.T
        #         )
        # else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rotation = self._rot_matrix_from_ang_mom()
        gas["Coordinates"] = rotation.apply(gas["Coordinates"])
        gas["Relative_Velocities"] = rotation.apply(gas["Relative_Velocities"])
        gas['Direction'] = rotation.apply(gas['Direction'])
        return gas

    def get_in_aperture(self, gas):
        """Select particles within the defined aperture radius."""
        in_aperture = gas["Physical_Relative_Distances"] < self.aperture_r
        return map_to_new_dict(gas, in_aperture)

    def get_outflow_mass(self, cold_only=False, in_aperture=False):
        """Return the total outflow mass, optionally limited to cold gas or a specific aperture."""
        gas = self.cold_out_gas if cold_only else self.out_gas
        if in_aperture:
            gas = self.get_in_aperture(gas)
        return np.sum(gas["Masses"])

    def get_average_outflow_vel(
        self, weighting="Masses", cold_only=False, in_aperture=False
    ):
        """Compute the weighted average outflow velocity."""
        gas = self.cold_out_gas if cold_only else self.out_gas
        if in_aperture:
            gas = self.get_in_aperture(gas)
        if weighting == "Luminosity":
            weights = gas["Density"] * gas["Masses"]
        elif weighting is None:
            weights = np.ones_like(gas["Flow_Velocities"])
        else:
            weights = gas[weighting]
        try:
            vel = np.abs(gas["Flow_Velocities"])
            return np.average(vel, weights=weights)
        except ZeroDivisionError:
            return None

    def _get_quantile_vout(self, gas, weights, quantile):
        """Return the quantile velocity of the outflow, weighted by the provided weights."""
        if len(gas["Flow_Velocities"]) < 3:
            return None
        vel = np.abs(gas["Flow_Velocities"])
        sorted_indices = np.argsort(vel)
        sorted_vel = vel[sorted_indices]
        sorted_weights = weights[sorted_indices]
        weights_cum = np.cumsum(sorted_weights)
        total_weight = weights_cum[-1]
        threshold = total_weight * quantile
        index = np.searchsorted(weights_cum, threshold, side="right")
        try:
            return sorted_vel[index]
        except IndexError:
            return None

    def get_quantile_velocity(
        self, quantile, weighting=None, cold_only=False, in_aperture=False
    ):
        """Compute a quantile velocity for the outflow."""
        gas = self.cold_out_gas if cold_only else self.out_gas
        if in_aperture:
            gas = self.get_in_aperture(gas)
        if weighting == "Flux":
            weights = gas["Masses"] * gas["Flow_Velocities"]
        elif weighting == "Masses":
            weights = gas["Masses"]
        elif weighting == "Luminosity":
            weights = gas["Density"] * gas["Masses"]
        elif weighting == "Luminosity_O3":
            weights = gas["Density"] * gas["Masses"] * gas["GFM_Metallicity"]
        elif weighting is None:
            weights = np.ones_like(gas["Flow_Velocities"])
        else:
            raise NotImplementedError
        try:
            return self._get_quantile_vout(gas, weights, quantile)
        except ZeroDivisionError:
            return None

    def get_flow_rate(self, cold_only=False, in_aperture=False):
        """Compute the flow rate from the selected outflowing gas."""
        if cold_only:
            gas = (
                self.get_in_aperture(self.cold_out_gas)
                if in_aperture
                else self.cold_out_gas
            )
        else:
            if in_aperture:
                gas = self.get_in_aperture(self.out_gas)
                r = self.aperture_r
            else:
                gas = self.out_gas
                if self.fixed_selection:
                    r = self.cut_r
                else:
                    r = self.cut_r / (self.z + 1) / 0.6774
        return np.sum(gas["Masses"] * gas["Flow_Velocities"] / r)

    def get_outflow_metallicity(self, cold_only=False, type="out", in_aperture=False):
        """Compute the mass-weighted outflow metallicity for either outflowing or remaining gas."""
        if type == "out":
            gas = (
                self.get_in_aperture(self.cold_out_gas)
                if cold_only
                else self.get_in_aperture(self.out_gas)
                if in_aperture
                else self.out_gas
            )
        elif type == "remain":
            gas = (
                self.get_in_aperture(self.remain_gas)
                if in_aperture
                else self.remain_gas
            )
        return float(np.average(gas["GFM_Metallicity"], weights=gas["Masses"]))

    def get_los_projections(self):
        self.gas["v_los_x"] = np.array(self.gas["Relative_Velocities"][:, 0], dtype=np.float32)
        self.gas["v_los_y"] = np.array(self.gas["Relative_Velocities"][:, 1], dtype=np.float32)
        self.gas["v_los_z"] = np.array(self.gas["Relative_Velocities"][:, 2], dtype=np.float32)
        return
