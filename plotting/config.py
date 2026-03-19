from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib import colormaps
import numpy as np
from astropy.cosmology import Planck18 as cosmo


def prop_labels(prop):
    _labels = {
        "Flow_Velocities": r"$v_\mathrm{r}$",
        "los_Velocities": r"$v_\mathrm{proj}$",
        "Masses": r"$\Sigma[\log(M/M_\odot)\mathrm{kpc}^{-2}]$",
        "StarFormationRate": r"$\Sigma_\mathrm{SFR}[\log(M_\odot)\mathrm{yr}^{-1}\mathrm{kpc}^{-2}]$",
        "Temperature": r"$T[\log(K)]$",
        "GFM_Metallicity": r"$\log(Z)$",
        "Rot_Velocities": r"$v_\mathrm{rot}$",
        "Angular_Velocities": r"$\omega_\mathrm{rot}$",
        "Galaxy_M_star": r"$M_\star[\log(M_\odot)]$",
        "Galaxy_SFR": r"SFR$[M_\odot/\mathrm{yr}]$",
        "M_star_log": r"$\log(M_\star/M_\odot)$",
        "v_lum": r"$\langle v \rangle_\mathrm{lum}[\mathrm{km}/\mathrm{s}]$",
        "v_lum_cold": r"$\langle v \rangle_\mathrm{lum}[\mathrm{km}/\mathrm{s}]$",
        "v_mass": r"$\langle v \rangle_\mathrm{M}[\mathrm{km}/\mathrm{s}]$",
        "v_mass_cold": r"$\langle v \rangle_\mathrm{M}[\mathrm{km}/\mathrm{s}]$",
        "M_out": r"$M_\mathrm{out}[M_\odot]$",
        "M_out_cold": r"$M_\mathrm{out}[M_\odot]$",
        "M_dot": r"$\dot{M}_\mathrm{out}[M_\odot\mathrm{km}/\mathrm{s}]$",
        "M_dot_cold": r"$\dot{M}_\mathrm{out}[M_\odot\mathrm{km}/\mathrm{s}]$",
        "cut_radius_abs": r"$r_\mathrm{cut}[\mathrm{kpc}]$",
        "v_lum_50": r"$v_{\mathrm{out}, 50\mathrm{}}[\mathrm{km}/\mathrm{s}]$",
        "v_lum_50_cold": r"$v_{\mathrm{out}, 50\mathrm{,cold}}[\mathrm{km}/\mathrm{s}]$",
        "v_lum_75": r"$v_{\mathrm{out}, 75\mathrm{}}[\mathrm{km}/\mathrm{s}]$",
        "v_lum_75_cold": r"$v_{\mathrm{out}, 75\mathrm{,cold}}[\mathrm{km}/\mathrm{s}]$",
        "v_lum_90": r"$v_{\mathrm{out}, 90\mathrm{}}[\mathrm{km}/\mathrm{s}]$",
        "v_lum_90_cold": r"$v_{\mathrm{out}, 90\mathrm{,cold}}[\mathrm{km}/\mathrm{s}]$",
        "v_mass_50": r"$v_{\mathrm{out}, 50\mathrm{}}[\mathrm{km}/\mathrm{s}]$",
        "v_mass_50_cold": r"$v_{\mathrm{out}, 50\mathrm{,cold}}[\mathrm{km}/\mathrm{s}]$",
        "v_mass_75": r"$v_{\mathrm{out}, 75\mathrm{}}[\mathrm{km}/\mathrm{s}]$",
        "v_mass_75_cold": r"$v_{\mathrm{out}, 75\mathrm{,cold}}[\mathrm{km}/\mathrm{s}]$",
        "v_mass_90": r"$v_{\mathrm{out}, 90\mathrm{}}[\mathrm{km}/\mathrm{s}]$",
        "v_mass_80": r"$v_{\mathrm{out}, 80\mathrm{}}[\mathrm{km}/\mathrm{s}]$",
        "v_mass_90_cold": r"$v_{\mathrm{out}, 90\mathrm{,cold}}[\mathrm{km}/\mathrm{s}]$",
        "v_mdot_50": r"$v_{\mathrm{out}, 50\mathrm{}}[\mathrm{km}/\mathrm{s}]$",
        "v_mdot_50_cold": r"$v_{\mathrm{out}, 50\mathrm{,cold}}[\mathrm{km}/\mathrm{s}]$",
        "v_mdot_75": r"$v_{\mathrm{out}, 75\mathrm{}}[\mathrm{km}/\mathrm{s}]$",
        "v_mdot_75_cold": r"$v_{\mathrm{out}, 75\mathrm{,cold}}[\mathrm{km}/\mathrm{s}]$",
        "v_mdot_90": r"$v_{\mathrm{out}, 90\mathrm{}}[\mathrm{km}/\mathrm{s}]$",
        "v_mdot_90_cold": r"$v_{\mathrm{out}, 90\mathrm{,cold}}[\mathrm{km}/\mathrm{s}]$",
        "SFR_log": r"$\mathrm{SFR}[\log(M_\odot/\mathrm{yr})]$",
        "fraction_lum": r"$f(L)$",
        "Relative_Velocities_abs": r"$|v|[\mathrm{km}/\mathrm{s}]$",
        "Luminosity": r"$L_{H\alpha}$[a.u.]",
        "Luminosity_light": r"$L_{H\alpha, \mathrm{dist}}$[a.u.]",
        "Luminosity_O3": r"$L_{OIII}$[a.u.]",
        "v_z": r"$v_z[\mathrm{km}/\mathrm{s}]$",
        "Distance": r"$M$[a.u.]",
        "M_out_log": r"$\log(M_\mathrm{out}/M_\odot)$",
        "M_out_aperture_log": r"$\log(M_\mathrm{out, 0.6''}/M_\odot)$",
        "M_out_and_wind_log": r"$\log(M_\mathrm{out, wind, 0.6''}/M_\odot)]$",
        "M_wind_log": r"$\log(M_\mathrm{wind, 0.6''}/M_\odot)$",
        "M_out_aperture_log_03": r"$M_\mathrm{out, 0.3''}[\log(M_\odot)]$",
        "M_gas_log": r"$M_\mathrm{gas}[\log(M_\odot)]$",
        "W80_galaxy": r"$W_{80, \mathrm{gal}}[\log( \mathrm{km}/\mathrm{s} )]$",
        "W80_outflow": r"$W_{80, \mathrm{out}}[\log( \mathrm{km}/\mathrm{s} )]$",
        "W_ratio": r"$W_{80, \mathrm{out}}/W_{80, \mathrm{gal}}$",
        "Z_ratio": r"$Z_\mathrm{out}/Z_\mathrm{gal}$",
        "Z_ratio_aperture": r"$Z_\mathrm{out, 0.6''}/Z_\mathrm{gal, 0.6''}$",
        "v_mass_aperture": r"$v_{\mathrm{out, 0.6''}}[\mathrm{km}/\mathrm{s}]$",
        "sfr_0_log": r"$\mathrm{SFR}_{0}[\log(M_\odot/\mathrm{yr})]$",
        "sfr_10_log": r"$\mathrm{SFR}_{10}[\log(M_\odot/\mathrm{yr})]$",
        "sfr_50_log": r"$\mathrm{SFR}_{50}[\log(M_\odot/\mathrm{yr})]$",
        "sfr_100_log": r"$\mathrm{SFR}_{100}[\log(M_\odot/\mathrm{yr})]$",
        "SFR_hist10_log": r"$\mathrm{SFR}_{10}[\log(M_\odot/\mathrm{yr})]$",
        "SFR_hist50_log": r"$\mathrm{SFR}_{50}[\log(M_\odot/\mathrm{yr})]$",
        "SFR_hist100_log": r"$\mathrm{SFR}_{100}[\log(M_\odot/\mathrm{yr})]$",
        "z": r"redshift",
        "sSFR_log": r"$\mathrm{sSFR}[\log(\mathrm{yr}^{-1})]$",
        "sSFR_log_100": r"$\mathrm{sSFR}_{100}[\log(\mathrm{yr}^{-1}]$",
        "sOutflow": r"log($M_\mathrm{out}/M_\star$)",
        "M_out/M_star": r"$M_\mathrm{out}/M_\star$",
        "sOutflow_lin": r"$M_\mathrm{out}/M_\star$",
        "lookback": r"lookback time [Gyr]",
        "BH_mdot_log": r"$\dot{M}_\mathrm{BH}[\log(M_\odot/\mathrm{yr})]$",
        "eta_log": r"$\log(\eta)$",
        "eta": r"$\eta$",
        "outflow_Z_log": r"$\log(Z_\mathrm{out})$",
        "v_los_x": r"$v_{\mathrm{los}, x}[\mathrm{km}/\mathrm{s}]$",
        "v_los_y": r"$v_{\mathrm{los}, y}[\mathrm{km}/\mathrm{s}]$",
        "v_los_z": r"$v_{\mathrm{los}, z}[\mathrm{km}/\mathrm{s}]$",
        "Relative_Distances": r"$d_\mathrm{rel}[\mathrm{kpc}]$",
    }
    return _labels[prop]


def get_ranges(prop, parameters):
    if prop == "Flow_Velocities":
        parameters["vmin"] = -150
        parameters["vcenter"] = 0
        parameters["vmax"] = 150

    elif prop == "los_Velocities":
        parameters["vmin"] = -250
        parameters["vcenter"] = 0
        parameters["vmax"] = 250

    elif prop == "Relative_Velocities_abs":
        parameters["vmin"] = -250
        parameters["vcenter"] = 0
        parameters["vmax"] = 250

    elif prop == "v_z":
        parameters["vmin"] = -250
        parameters["vcenter"] = 0
        parameters["vmax"] = 250

    elif "v_los" in prop:
        parameters["vmin"] = -300
        parameters["vcenter"] = 0
        parameters["vmax"] = 300

    elif prop == "Rot_Velocities":
        parameters["vmin"] = 0
        parameters["vcenter"] = 1500
        parameters["vmax"] = 3000

    elif prop == "Angular_Velocities":
        parameters["vmin"] = 0
        parameters["vcenter"] = 300
        parameters["vmax"] = 600

    elif prop == "Masses":
        parameters["vmin"] = 6.0
        parameters["vcenter"] = 7.5
        parameters["vmax"] = 9

    elif prop == "GFM_Metallicity":
        parameters["vmin"] = -3.5
        parameters["vcenter"] = -2.5
        parameters["vmax"] = -1.5

    elif prop == "Temperature":
        parameters["vmin"] = 4
        parameters["vcenter"] = 6.0
        parameters["vmax"] = 8

    elif prop == "StarFormationRate":
        parameters["vmin"] = -3
        parameters["vcenter"] = 0
        parameters["vmax"] = 3

    elif prop == "sfr_100_log":
        parameters["vmin"] = -3
        parameters["vcenter"] = 0
        parameters["vmax"] = 3

    elif prop == "Relative_Distances":
        parameters["vmin"] = 0
        parameters["vcenter"] = 2
        parameters["vmax"] = 4
    return


def plot_parameters_comp(prop=None):
    parameters = {}

    parameters["titlesize"] = 30
    parameters["label_fontsize"] = 30

    parameters["colorbar_labelsize"] = 25
    parameters["colorbar_ticklabelsize"] = 15
    parameters["ticklabelsize"] = 20

    parameters["height_per_image"] = 6
    parameters["width_per_image"] = 6

    if prop is not None:
        get_ranges(prop, parameters)

    return parameters


def get_cmap(prop):
    coolwarm_props = {
        "Flow_Velocities",
        "Rot_Velocities",
        "Angular_Velocities",
        "los_Velocities",
        "v_z",
    }
    bwr_props = {"v_los_x", "v_los_y", "v_los_z"}
    if prop in coolwarm_props:
        hue_neg, hue_pos = 250, 15
        colormap = sns.diverging_palette(
            hue_neg, hue_pos, center="dark", as_cmap=True
        )
    elif prop in bwr_props:
        colormap = colormaps["bwr"]
    else:
        cmap = "inferno"
        colormap = colormaps[cmap]
    return colormap


def label_colors(for_slides):
    if for_slides:
        params = {
            "ytick.color": "w",
            "xtick.color": "w",
            "axes.labelcolor": "w",
            "axes.edgecolor": "w",
            "axes.facecolor": "black",
            "legend.labelcolor": "w",
            "axes.titlecolor": "w",
        }
        plt.rcParams.update(params)
    else:
        params = {
            "ytick.color": "black",
            "xtick.color": "black",
            "axes.labelcolor": "black",
            "axes.edgecolor": "black",
            "axes.facecolor": "white",
            "legend.labelcolor": "black",
            "axes.titlecolor": "black",
        }
        plt.rcParams.update(params)
    return


def get_universe_age(redshift):
    age = cosmo.age(redshift).value  # in Gyr
    return age
