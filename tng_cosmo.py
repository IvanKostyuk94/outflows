"""
IllustrisTNG cosmology based on astropy.

Replaces pyTNG.cosmology.TNGcosmo.
See http://www.tng-project.org/w/index.php/TNG_Simulation_Series#Cosmology
"""
from astropy.cosmology import Planck15

TNGcosmo = Planck15.clone(
    name="TNG cosmology",
    H0=100 * 0.6774,
    Om0=0.3089,
    Ob0=0.0486,
)
