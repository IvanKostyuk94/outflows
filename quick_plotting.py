from plotting import *
import pandas as pd
import numpy as np
from astropy.cosmology import Planck18 as cosmo

serra_path = "/ptmp/mpa/ivkos/outflows/serra_out_W80_sfr.hdf5"
# path = "/ptmp/mpa/ivkos/outflows/in_aperture_final.hdf5"
path = "/ptmp/mpa/ivkos/outflows/in_aperture_wind.hdf5"

# path = "/ptmp/mpa/ivkos/outflows/random_projections.hdf5"
df = pd.read_hdf(path)
df_serra = pd.read_hdf(serra_path)
df['M_out_log'] = np.log10(df['M_out']*1e10/cosmo.h)
df['M_out_aperture_log'] = np.log10(df['M_out_0.6']*1e10/cosmo.h)
df['M_out_aperture_log_03'] = np.log10(df['M_out_aperture']*1e10/cosmo.h)
df['M_star_log'] = np.log10(df['Galaxy_M_star']*1e10/cosmo.h)
df['SFR_hist10_log'] = np.log10(df['SFR_hist10'])
df['Z_ratio'] = df['outflow_Z_aperture']/df['remain_Z_aperture']
df_serra['M_out_aperture_log'] = np.log10(df_serra['M_out_0.6'])
df_serra['SFR_hist10_log'] = np.log10(df_serra['sfr_10'])
df_serra['Z_ratio'] = df_serra['outflow_Z_aperture']/df_serra['remain_Z_aperture']

filter1 = df.M_star_log < 8
title1 = r"$M_\star < 10^8M_\odot$"
filter2 = (df.M_star_log>8.5)&(df.M_star_log<9)
title2 = r"$10^{8.5}M_\odot<M_\star < 10^9M_\odot$"
filter3 = df.M_star_log>9
title3 = r"$M_\star > 10^9M_\odot$"
title_tng = "TNG50"
title_serra = "SERRA"

# plot_W80_evolution(
#     df_serra,
#     theta_angles=[0,90],
#     phi_angles=[0],
#     bins=100,
#     cumulative=True,
#     for_slides=True,
#     title=title_serra,
#     aperture=False,
# )
from serra_backend import SerraBackend
_serra_backend = SerraBackend(base_path="/ptmp/mpa/ivkos/outflows")

plot_prop_maps_grouped(
    halo_id=3522,
    df=df_serra,
    snap=62,
    props=['Masses'],
    backend=_serra_backend,
    grid_size=100,
    method="GMM",
    group_props=None,
    dirs=[1, 2],
    sizebar_length=1,
    projection_angle_theta=None,
    projection_angle_phi=0,
    for_slides=True,
)