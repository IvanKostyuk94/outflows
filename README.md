# outflows

A Python toolkit for measuring galactic outflows in cosmological simulations, with support for IllustrisTNG and other hydrodynamical simulation backends.

## What it does

For each galaxy in a simulation snapshot, the code identifies outflowing gas particles — those with radial velocities exceeding the local escape velocity — and computes:

- **Outflow mass** within a spherical aperture (default 0.6 × R<sub>200</sub>)
- **Mass-weighted velocity quantiles** (v<sub>50</sub>, v<sub>80</sub>, v<sub>90</sub>)
- **Outflow and remaining gas metallicities**
- **Line-of-sight kinematics**: W80 velocity width and velocity offset Δv at multiple projection angles
- **Wind particle masses** (where available from the simulation)

Three complementary approaches are implemented: a particle-based galaxy class (`Galaxy`), a spherical-shell method (`GalaxyShells`), and a line-of-sight projection class (`GalaxyProjections`). Results are written back into an HDF5 galaxy catalogue.

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
| `cutout_scale` | Radius multiplier around R₂₀₀ for particle cutouts |

## Usage

```python
from tng_backend import TNGBackend
from analyse_outflow_properties import OutflowPropUpdater
from config import config

backend = TNGBackend(config=config)

updater = OutflowPropUpdater(
    df_name="my_galaxy_catalogue",
    backend=backend,
    snap_range=[13, 26],   # snapshot range to process
    in_aperture=True,
    aperture_size=0.6,     # fraction of R200
    with_quantile=True,    # compute velocity quantiles instead of total mass
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
├── gaussian_outflow_selection.py  # Gaussian decomposition for outflow/disc separation
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
├── serra_backend.py               # Serra simulation backend
├── tng_cosmo.py                   # TNG cosmological parameters
├── utils.py                       # Shared helper functions
│
├── plotting.py                    # Full plotting library for outflow results
├── quick_plotting.py              # Quick diagnostic plots
├── plot_progenitor_evolution.py   # Progenitor evolutionary tracks
├── plotting/                      # Output directory for figures
│
├── tests/                         # Unit tests (pytest)
├── testing.ipynb                  # Interactive exploration notebook
└── deprecated/                    # Old scripts retained for reference
```
