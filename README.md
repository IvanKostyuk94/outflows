# outflows

Code accompanying the paper:

> **High-Redshift Galactic Outflows: Orientation Effects, Kinematics, and Metallicity in TNG50 and SERRA**
> Ivan Kostyuk, Stefano Carniani, Mahsa Kohandel, Andrea Pallottini
> *Submitted to A&A* — [arXiv:2512.08543](https://arxiv.org/abs/2512.08543)

## Science context

JWST/NIRSpec has provided the first detections of warm ionised outflows in low-mass, high-redshift galaxies (z > 3), with an observed occurrence rate of 25–40%. This is lower than predicted by simulations, which suggest that fast outflowing gas should be nearly ubiquitous in early star-forming galaxies. This code was written to understand that discrepancy.

We identify and characterise galactic outflows in ~60 000 galaxies from [TNG50](https://www.tng-project.org/) and ~3 000 galaxies from the [SERRA](https://andrea-pallottini.com/research/serra/) zoom-in suite, spanning redshifts z = 3–5 and stellar masses M<sub>★</sub> = 10<sup>7.5</sup>–10<sup>11</sup> M<sub>☉</sub>. The analysis focuses on three questions:

1. **Detectability vs. orientation** — are outflows more easily seen face-on or edge-on?
2. **Kinematics** — how do simulated outflow velocities compare to JWST/JADES measurements?
3. **Metallicity** — what is the metal content of the outflowing vs. remaining gas?

Key results: outflow masses broadly reproduce JWST/JADES observations within ~0.5 dex, but simulated outflow velocities are typically an order of magnitude lower. TNG50 shows a clear orientation dependence: face-on galaxies are ~15% more likely to show a detectable outflow than edge-on ones, rising to ~40% for massive disc-shaped systems.

## Method

Outflowing gas particles are identified using a **Gaussian mixture model** (GMM) that takes gas radial velocity, SFR surface density, and galactocentric distance as inputs. This separates the outflowing component from the disc and circumgalactic medium without an arbitrary velocity threshold. Line-of-sight projections at multiple angles then mock-observe each galaxy to quantify orientation effects.

## What the code computes

For each galaxy in the catalogue the code measures:

- **Outflow mass** within a spherical aperture (default 0.6 × R<sub>200</sub>)
- **Mass-weighted velocity quantiles** (v<sub>50</sub>, v<sub>80</sub>, v<sub>90</sub>)
- **Outflow and remaining gas metallicities**
- **Line-of-sight kinematics**: W80 velocity width and velocity offset Δv at multiple projection angles
- **Wind particle masses** (TNG wind model)

## Installation

```bash
git clone https://github.com/IvanKostyuk94/outflows.git
cd outflows
pip install .
```

**Optional extras:**
```bash
pip install ".[tng]"   # adds illustris_python for reading TNG data directly
pip install ".[dev]"   # adds pytest
```

**Requirements:** Python ≥ 3.9, NumPy, SciPy, pandas, astropy, h5py, numba, PyYAML, scikit-learn, tables.

## Configuration

Edit `config_parameters.yml` before running:

| Parameter | Description |
|---|---|
| `base_path` | Root directory for output HDF5 catalogues |
| `tng_datapath` | Path to IllustrisTNG simulation data |
| `sim_name` | Simulation suite (e.g. `L35n2160TNG`) |
| `cutout_scale` | Radius multiplier around R<sub>200</sub> for particle cutouts |

## Usage

```python
from tng_backend import TNGBackend
from analyse_outflow_properties import OutflowPropUpdater
from config import config

backend = TNGBackend(config=config)

updater = OutflowPropUpdater(
    df_name="my_galaxy_catalogue",
    backend=backend,
    snap_range=[13, 26],   # snapshot range (TNG50 z=3-5)
    in_aperture=True,
    aperture_size=0.6,     # fraction of R200
    with_quantile=True,    # compute velocity quantiles
)

updater.add_outflow_parameters()
updater.save_df()
```

## Repository layout

```
outflows/
├── config.py                      # Loads config_parameters.yml
├── config_parameters.yml          # All user-facing parameters
│
├── build_galaxy_db.py             # Builds the initial galaxy HDF5 catalogue from TNG
├── write_halo_db.py               # Writes halo-level properties to the database
├── process_gas.py                 # Core Galaxy class: particle selection and outflow identification
├── galaxy_shell_outflows.py       # Shell-based outflow measurement (GalaxyShells)
├── gaussian_outflow_selection.py  # GMM-based outflow/disc separation
├── los_projection.py              # Line-of-sight projection and W80 kinematics
├── random_projection.py           # Random-orientation projection averaging
├── Grid_halo.py                   # SPH-smoothed grid maps per halo
├── sph_gridding.py                # SPH kernel gridding utilities
│
├── analyse_outflow_properties.py  # OutflowPropUpdater: adds outflow columns to catalogue
├── add_metallicities.py           # Enriches catalogue with gas metallicities
├── add_sfr_radius.py              # Adds SFR-weighted half-mass radius
├── add_tng_sfr_hist.py            # Appends SFR history from TNG merger trees
├── find_progenitors.py            # Traces galaxy progenitors across snapshots
├── wind_mass_updater.py           # Adds wind particle masses (TNG wind model)
├── convergence_analyser.py        # Resolution convergence diagnostics
│
├── backends.py                    # Abstract backend interface
├── tng_backend.py                 # IllustrisTNG data backend
├── serra_backend.py               # SERRA simulation backend
├── tng_cosmo.py                   # TNG cosmological parameters
├── utils.py                       # Shared helper functions
│
├── plotting.py                    # Full plotting library for paper figures
├── quick_plotting.py              # Quick diagnostic plots
├── plot_progenitor_evolution.py   # Progenitor evolutionary tracks
├── plotting/                      # Output directory for figures
│
├── tests/                         # Unit tests (pytest)
├── testing.ipynb                  # Interactive exploration notebook
└── deprecated/                    # Old scripts retained for reference
```

## Citation

If you use this code, please cite:

```bibtex
@article{Kostyuk2025outflows,
  author  = {Kostyuk, Ivan and Carniani, Stefano and Kohandel, Mahsa and Pallottini, Andrea},
  title   = {High-Redshift Galactic Outflows: Orientation Effects, Kinematics, and Metallicity in {TNG50} and {SERRA}},
  journal = {arXiv},
  year    = {2025},
  eprint  = {2512.08543},
  archivePrefix = {arXiv},
  primaryClass  = {astro-ph.GA}
}
```
