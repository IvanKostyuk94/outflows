"""Abstract base class for simulation backends."""

from abc import ABC, abstractmethod
import numpy as np


class SimulationBackend(ABC):
    """Interface that each simulation type (TNG, Serra, etc.) must implement.

    The Galaxy class delegates all simulation-specific logic to the backend,
    eliminating conditional branching based on simulation type.
    """

    @abstractmethod
    def load_gas(self, snap, halo_id, galaxy_id=None):
        """Load gas particles. Returns dict with standardized keys:
        Coordinates, Velocities, Masses, Density, StarFormationRate,
        GFM_Metallicity, count. May also include InternalEnergy,
        ElectronAbundance, hsml depending on simulation."""

    @abstractmethod
    def load_stars(self, snap, halo_id, galaxy_id=None):
        """Load star particles. Returns dict with:
        Coordinates, Velocities, Masses, count."""

    @abstractmethod
    def load_dm(self, snap, halo_id):
        """Load dark matter particles (for escape velocity calculations)."""

    @abstractmethod
    def load_halo_stars(self, snap, halo_id):
        """Load all stars in the halo (not just subhalo)."""

    @abstractmethod
    def get_redshift(self, snap, halo_row=None):
        """Return redshift for a given snapshot."""

    @abstractmethod
    def get_halo_id_column(self):
        """Column name for halo identification in the galaxy DataFrame."""

    @abstractmethod
    def get_galaxy_id(self, halo_row):
        """Extract the galaxy/subhalo ID from a DataFrame row."""

    # --- Flags controlling Galaxy class behavior ---

    @abstractmethod
    def needs_coordinate_offset(self):
        """Whether particle coordinates need to be shifted by galaxy position."""

    @abstractmethod
    def needs_velocity_scaling(self):
        """Whether velocities need sqrt(a) scaling (Arepo convention)."""

    @abstractmethod
    def needs_density_conversion(self):
        """Whether density needs electron-density conversion."""

    @abstractmethod
    def needs_temperature_computation(self):
        """Whether temperature must be computed from internal energy."""

    @abstractmethod
    def needs_hsml_computation(self):
        """Whether smoothing length must be computed from mass/density."""

    @abstractmethod
    def has_virial_radius(self):
        """Whether R_vir is available in the galaxy DataFrame."""

    @abstractmethod
    def has_sfr_dist(self):
        """Whether SFR_dist should be computed for particles."""

    @abstractmethod
    def has_wind_particles(self):
        """Whether wind particles exist (TNG star particles with negative formation time)."""

    def get_mean_velocity_weights(self, particles):
        """Return weights for mean velocity computation.
        None means simple (unweighted) average. Override to return
        a mass array for mass-weighted averaging of inner particles."""
        return None

    def gmm_mass_weighted(self):
        """Whether GMM fitting should use mass-weighted resampling."""
        return False

    def gmm_distance_weighted(self):
        """Whether galaxy group selection uses mass-weighted distance."""
        return False
