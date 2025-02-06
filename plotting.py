from matplotlib import pyplot as plt
from matplotlib import colors
import seaborn as sns
from matplotlib import colormaps
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from utils import get_redshift
from Grid_halo import GasGridder
from process_gas import Galaxy
from los_projection import GalaxyProjections
from utils import get_halo, get_redshift
import scipy
from scipy import stats
from functools import partial


def prop_labels(prop):
    prop_labels = {
        "Flow_Velocities": r"$v_\mathrm{out}$",
        "los_Velocities": r"$v_\mathrm{proj}$",
        "Masses": r"$\Sigma[\log(M_\odot)\mathrm{kpc}^{-2}]$",
        # "Masses": r"$M[\mathrm{a.u.}]$",
        "StarFormationRate": r"$\Sigma_\mathrm{SFR}[\log(M_\odot)\mathrm{yr}^{-1}\mathrm{kpc}^{-2}]$",
        "Temperature": r"$T[\log(K)]$",
        "GFM_Metallicity": r"$\log(Z)$",
        "Rot_Velocities": r"$v_\mathrm{rot}$",
        "Angular_Velocities": r"$\omega_\mathrm{rot}$",
        "Galaxy_M_star": r"$M_\star[\log(M_\odot)]$",
        "Galaxy_SFR": r"SFR$[M_\odot/\mathrm{yr}]$",
        "M_star_log": r"$M_\star[\log(M_\odot)]$",
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
        "SFR_log": r"SFR$[\log(M_\odot/\mathrm{yr})]$",
        "fraction_lum": r"$f(L)$",
        "Relative_Velocities_abs": r"$|v|[\mathrm{km}/\mathrm{s}]$",
        "Luminosity": r"$L_{H\alpha}$[a.u.]",
        "Luminosity_light": r"$L_{H\alpha, \mathrm{dist}}$[a.u.]",
        "Luminosity_O3": r"$L_{OIII}$[a.u.]",
        "v_z": r"$v_z[\mathrm{km}/\mathrm{s}]$",
        "Distance": r"$M$[a.u.]",
        "M_out_log": r"$M_\mathrm{out}[\log(M_\odot)]$",
        "M_out_aperture_log": r"$M_\mathrm{out, 0.6''}[\log(M_\odot)]$",
        "M_out_aperture_log_03": r"$M_\mathrm{out, 0.3''}[\log(M_\odot)]$",
        "M_gas_log": r"$M_\mathrm{gas}[\log(M_\odot)]$",
        "SFR_log": r"$\mathrm{SFR}[\log(M_\odot/\mathrm{yr})]$",
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
        "sSFR_log": r"$\mathrm{sSFR}[\log(\mathrm{yr}^{-1}]$",
        "sSFR_log_100": r"$\mathrm{sSFR}_{100}[\log(\mathrm{yr}^{-1}]$",
        "sOutflow": r"log($M_\mathrm{out}/M_\star$)",
        "lookback": r"lookback time [Gyr]",
        "BH_mdot_log": r'$\dot{M}_\mathrm{BH}[\log(M_\odot/\mathrm{Gyr})]$',
        "eta_log": r'$\log(\eta)$',
        "eta":r'$\eta$',
    }
    return prop_labels[prop]


def get_ranges(prop, parameters):
    # if prop == "Flow_Velocities":
    #     parameters["vmin"] = 50
    #     parameters["vcenter"] = 150
    #     parameters["vmax"] = 250
    if prop == "Flow_Velocities":
        parameters["vmin"] = -110
        parameters["vcenter"] = 0
        parameters["vmax"] = 110

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

    elif prop == "Rot_Velocities":
        parameters["vmin"] = 0
        parameters["vcenter"] = 1500
        parameters["vmax"] = 3000

    elif prop == "Angular_Velocities":
        parameters["vmin"] = 0
        parameters["vcenter"] = 300
        parameters["vmax"] = 600

    # elif prop == "Masses":
    #     parameters["vmin"] = 7.0
    #     parameters["vcenter"] = 8.0
    #     parameters["vmax"] = 9.0

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
    if prop in coolwarm_props:
        hue_neg, hue_pos = 250, 15
        colormap = sns.diverging_palette(hue_neg, hue_pos, center="dark", as_cmap=True)
        # cmap = "coolwarm"
    else:
        cmap = "inferno"
        colormap = colormaps[cmap]
    return colormap


def create_color_bar(
    f,
    ax,
    parameters,
    subfig,
    prop,
    label=None,
    ax_is_cbar=False,
    horizontal=False,
    gap=False,
):
    if ax_is_cbar:
        if horizontal:
            cbar = f.colorbar(subfig, cax=ax, orientation="horizontal")
        else:
            cbar = f.colorbar(subfig, cax=ax)

    else:
        divider = make_axes_locatable(ax)
        if gap:
            pad = 10
        else:
            pad = 0.15
        cax = divider.append_axes("right", size="5%", pad=pad)
        cbar = f.colorbar(subfig, cax=cax)

    if label is not None:
        size = parameters["colorbar_labelsize"]
        cbar.set_label(label, size=size, labelpad=18)

    ticksize = parameters["colorbar_ticklabelsize"]
    cbar.ax.tick_params(labelsize=ticksize)

    if prop == "Flow_Velocities":
        cbar.ax.set_yticks([-100, -50, 0, 50, 100, 100])  # Set tick positions
        cbar.ax.set_yticklabels([-100, -50, 0, 50, 100, 100], fontsize=ticksize)
    return


def draw_sizebar(ax, gridder, length_kpc=1):
    pixel_length_abs = gridder.get_pixel_length_abs()
    length_bar = length_kpc / pixel_length_abs
    asb = AnchoredSizeBar(
        ax.transData,
        length_bar,
        f"{length_kpc}kpc",
        size_vertical=0.5,
        loc="lower right",
        pad=0.5,
        borderpad=0.5,
        sep=5,
        frameon=False,
        color="white",
    )
    ax.add_artist(asb)
    return


def get_col_norm(parameters):
    col_norm = colors.TwoSlopeNorm(
        vmin=parameters["vmin"],
        vcenter=parameters["vcenter"],
        vmax=parameters["vmax"],
    )
    return col_norm


def setup_prop_parameters(parameters, columns, rows):
    figsize = (
        parameters["width_per_image"] * (columns - 11 / 12),
        parameters["height_per_image"] * rows,
    )
    width_ratios = [12 for _ in range(columns - 1)]
    width_ratios.append(1)
    fig, axs = plt.subplots(
        ncols=columns,
        nrows=rows,
        gridspec_kw={
            "wspace": 0.05,
            "hspace": 0.07,
            "width_ratios": width_ratios,
        },
        figsize=figsize,
    )
    return fig, axs


def plot_prop_maps(gridder, prop, dirs, sizebar_length=1):
    parameters = plot_parameters_comp(prop)

    columns = len(dirs) + 1
    rows = len(gridder.grids)
    fig, axs = setup_prop_parameters(parameters, columns, rows)

    fontsize = parameters["label_fontsize"]
    axs[0, 0].set_title("Edge on", fontsize=fontsize)
    axs[0, 1].set_title("Face on", fontsize=fontsize)
    gas_types = ["all", "outflow", "remain"]
    for row in range(rows):
        for column in range(columns):
            ax = axs[row, column]
            if column == 0:
                # if prop == "Flow_Velocities":
                #     color='black'
                # else:
                #     color='white'
                color = "white"
                ax.text(2, 93, gas_types[row], fontsize=20, color=color)
            cmap = get_cmap(prop=prop)

            if column < (columns - 1):
                image = gridder.get_prop_image(
                    row,
                    prop,
                    dir=dirs[column],
                )
                subfig = ax.pcolormesh(
                    image,
                    cmap=cmap,
                    norm=get_col_norm(parameters),
                )
                draw_sizebar(ax, gridder, length_kpc=sizebar_length)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

            else:
                create_color_bar(
                    fig,
                    ax,
                    parameters,
                    subfig,
                    prop=prop,
                    label=prop_labels(prop),
                    ax_is_cbar=True,
                    horizontal=False,
                )
    return


def get_data(df, prop_x, prop_y, bins, by_z=False):
    log_properties = {"M_out", "M_dot", "M_out_cold", "M_dot_cold"}
    df = df[~np.isnan(df[prop_y])]
    df = df[(df.snap < 26) & (df.snap > 16)]
    df["bin"] = pd.cut(df[prop_x], bins=bins)
    x_centers = df["bin"].apply(lambda x: x.mid).unique()
    sorted_indices = np.argsort(x_centers)
    x_centers = x_centers[sorted_indices]
    y_means_all = []
    y_errors_all = []
    labels = []
    if by_z:
        for i in range(17, 26):
            sub_df = df[df.snap == i]
            z = get_redshift(i)
            label = f"z = {z:.1f}"
            if prop_y in log_properties:
                values = np.log10(sub_df.groupby("bin")[prop_y])
            else:
                values = sub_df.groupby("bin")[prop_y]
            y_means = values.quantile(0.75)
            y_means = np.array(y_means)
            # y_means = values.mean()
            y_errors = values.sem()
            y_means_all.append(y_means)
            y_errors_all.append(y_errors)
            labels.append(label)
    else:
        label = "z = 3-5"
        if prop_y in log_properties:
            values = np.log10(df.groupby("bin")[prop_y])
        else:
            values = df.groupby("bin")[prop_y]

        y_means = values.quantile(q=0.5)
        y_errors = values.sem()
        y_means_all.append(y_means)
        y_errors_all.append(y_errors)
        labels.append(label)
    return x_centers, y_means_all, y_errors_all, labels


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


def plot_prop_correlation(
    df, prop_x, prop_y, bins=20, by_z=False, stepsize=1, for_slides=False
):
    label_colors(for_slides)
    if prop_x == "SFR_log":
        df = df[df.SFR_log > -5]
    if prop_x == "M_star_log":
        df = df[df.M_star_log > 7.5]
    x_centers, y_means_all, y_errors_all, labels = get_data(
        df=df, prop_x=prop_x, prop_y=prop_y, bins=bins, by_z=by_z
    )
    fig, ax = plt.subplots(figsize=(15, 10))
    if not by_z:
        ax.scatter(df[prop_x], df[prop_y], s=1, alpha=0.3, color="red")
    for i in range(0, len(y_means_all), stepsize):
        ax.plot(
            x_centers,
            y_means_all[i],
            linewidth=3,
            label=labels[i],
        )

    parameters = plot_parameters_comp()
    ax.set_xlabel(prop_labels(prop_x), fontsize=parameters["label_fontsize"])
    ax.set_ylabel(prop_labels(prop_y), fontsize=parameters["label_fontsize"])
    ax.tick_params(labelsize=parameters["ticklabelsize"])
    ax.set_ylim(0, 600)
    ax.legend(fontsize=15)
    return


def get_weights(gas, weighting=None):
    if weighting == "Luminosity":
        weights = (gas["Density"] * gas["Masses"]) / gas["Density"].mean()
    elif weighting == "Luminosity_light":
        weights = gas["Density"] * gas["Masses"] / gas["SFR_dist"] ** 2
        test = gas["Density"] / gas["SFR_dist"] ** 2
        weights = gas["Density"] * gas["Masses"] / gas["SFR_dist"] ** 2 / test.mean()
    elif weighting == "Luminosity_O3":
        weights = gas["Density"] * gas["Masses"] * gas["GFM_Metallicity"]
    elif weighting == "Distance":
        weights = gas["Masses"] / gas["SFR_dist"] ** 2
    elif weighting is None:
        weights = np.ones_like(gas["Flow_Velocities"])
    else:
        try:
            weights = gas[weighting]
        except KeyError:
            raise KeyError(f"weighting {weighting} is not been implemented")
    return weights


def get_histogram(gas, velocity_type, bins, weights):
    heights, _ = np.histogram(
        gas[velocity_type],
        bins=bins,
        # density=True,
        weights=weights,
    )
    return heights


def Gauss(x, a, x0, sigma):
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2))


def Gauss2(x, a, x0_1, sigma, delta_a, x0_2, delta_sigma):
    return (a * np.exp(-((x - x0_1) ** 2) / (2 * sigma**2))) + (
        a * delta_a * np.exp(-((x - x0_2) ** 2) / (2 * (sigma * delta_sigma) ** 2))
    )


def get_reduced_chi_squared(x, y, y_err, model, popt):
    chi_squared = np.sum((y - model(x, *popt)) ** 2 / y_err**2)
    dof = len(x) - len(popt)
    return (chi_squared / dof,)


def plot_velocity_histogram(
    gases,
    velocity_types,
    weighting="Luminosity",
    bin_n=80,
    range=None,
    labels=None,
    norms=None,
    for_slides=False,
    title=None,
):
    label_colors(for_slides)
    if norms is None:
        norms = np.ones(len(gases))
    fig, ax = plt.subplots(figsize=(15, 10))
    if range is None:
        # bins = np.linspace(
        #     gases[0][velocity_types[0]].min(),
        #     gases[0][velocity_types[0]].max(),
        #     bin_n,
        # )
        bins = np.linspace(-1200, 1200, 80)

    else:
        bins = np.linspace(range[0], range[1], bin_n + 1)
    centers = (bins[1:] + bins[:-1]) / 2
    widths = bins[1:] - bins[:-1]
    tot_heights = np.zeros_like(centers)
    for i, gas in enumerate(gases):
        if i > 0:
            weights = get_weights(gas, weighting=weighting[i])
            heights = get_histogram(gas, velocity_types[i], bins, weights)
            if i == 1:
                ax.bar(
                    centers,
                    heights,
                    width=widths,
                    label=labels[i],
                    alpha=0.3,
                )
            tot_heights = tot_heights + heights
    ax.bar(centers, tot_heights, width=widths, label="summed", alpha=0.3)
    # try:
    bounds_1 = ([0, -np.inf, 0], [np.inf, np.inf, 150])
    popt, pcov = scipy.optimize.curve_fit(
        Gauss, centers, heights, p0=[max(heights), 0, 10], bounds=bounds_1
    )
    bounds_2 = (
        [0, -np.inf, 0, 0, -np.inf, 1.2],
        [np.inf, np.inf, 150, 0.5, np.inf, np.inf],
    )
    popt2, pcov2 = scipy.optimize.curve_fit(
        Gauss2, centers, heights, p0=[max(heights), 0, 30, 0.3, 0, 2], bounds=bounds_2
    )
    pop_gauss1 = popt2[:3]
    pop_gauss2 = [popt2[0] * popt2[3], popt2[4], popt2[5] * popt2[2]]
    ax.plot(centers, Gauss(centers, *popt), "r-", label="fit Gauss")
    ax.plot(centers, Gauss(centers, *pop_gauss1), "b-", label="fit 2 Gauss 1")
    ax.plot(centers, Gauss(centers, *pop_gauss2), "g-", label="fit 2 Gauss 2")
    # except:
    # pass

    parameters = plot_parameters_comp()
    ax.set_xlabel(
        prop_labels(velocity_types[-1]), fontsize=parameters["label_fontsize"]
    )
    ax.set_ylabel(prop_labels(weighting[i]), fontsize=parameters["label_fontsize"])
    ax.tick_params(labelsize=parameters["ticklabelsize"])

    ax.legend(fontsize=15)
    # ax.set_yscale("log")
    if title is not None:
        ax.set_title(title, fontsize=25)
    return


def plot_density_histogram(
    gas,
    bin_n=20,
    for_slides=False,
    title=None,
):
    label_colors(for_slides)
    fig, ax = plt.subplots(figsize=(15, 10))
    quantity = np.log10(gas["Density_e"])
    bins = np.linspace(
        quantity.min(),
        quantity.max(),
        bin_n,
    )
    centers = (bins[1:] + bins[:-1]) / 2
    heights, _ = np.histogram(
        quantity,
        bins=bins,
    )

    ax.bar(
        centers,
        heights,
    )

    parameters = plot_parameters_comp()
    ax.set_xlabel(r"$n_e[\mathrm{cm}^{-3}]$", fontsize=parameters["label_fontsize"])
    ax.set_ylabel(
        r"$\log(n_\mathrm{galaxies}$",
        fontsize=parameters["label_fontsize"],
    )
    ax.tick_params(labelsize=parameters["ticklabelsize"])
    ax.legend(fontsize=15)
    # ax.set_yscale("log")
    if title is not None:
        ax.set_title(title, fontsize=25)
    return


def plot_los_histograms(halo_id, snap, df, angles_theta, angles_phi, bin_n=100):
    columns = len(angles_theta)
    rows = len(angles_phi)
    figsize = (5 * columns, 5 * rows)
    sample = get_halo(df, snap, halo_id)
    # title = rf"$M_\star = 10^{{{sample.M_star_log:.1f}}}M_\odot, z={get_redshift(snap):.1f}$"
    fig, axs = plt.subplots(
        ncols=columns,
        nrows=rows,
        gridspec_kw={
            "wspace": 0.13,
            "hspace": 0.16,
        },
        figsize=figsize,
    )
    for i, theta in enumerate(angles_theta):
        for j, phi in enumerate(angles_phi):
            gal = GalaxyProjections(
                df=df,
                halo_id=int(halo_id),
                snap=int(snap),
                projection_angle_theta=theta,
                projection_angle_phi=phi,
            )

            gal.project_outflows()
            # gal.use_only_warm()

            gases = []
            labels = []
            velocity_types = []
            norms = []
            gases.append(gal.gas)
            gases.append(gal.out_gas)
            gases.append(gal.remain_gas)
            labels = ["all", "out", "remain"]

            velocity_types.append("los_Velocities")
            velocity_types.append("los_Velocities")
            velocity_types.append("los_Velocities")
            bins = np.linspace(
                gases[0][velocity_types[0]].min(),
                gases[0][velocity_types[0]].max(),
                bin_n,
            )
            centers = (bins[1:] + bins[:-1]) / 2
            widths = bins[1:] - bins[:-1]
            for k, gas in enumerate(gases):
                weights = get_weights(gas)
                heights = get_histogram(gas, velocity_types[i], bins, weights)
                v_range = np.quantile(gas["los_Velocities"], [0.1, 0.9])
                W80 = v_range[1] - v_range[0]

                axs[i, j].bar(
                    centers,
                    heights,
                    width=widths,
                    label=labels[k] + f", W80={W80:.1f}",
                    alpha=0.3,
                )
            title = rf"$\theta={theta:.1f}, \phi={phi:.1f}$"
            axs[i, j].set_title(title)
            axs[i, j].legend()
    return


def create_color_bar_hist(
    f,
    ax,
    subfig,
    label=None,
    multiple=False,
    horizontal=False,
    gap=False,
    prop=None,
):

    orientation = "horizontal" if horizontal else "vertical"
    cbar = f.colorbar(subfig, cax=ax, orientation=orientation)

    size = 25
    ticksize = 15

    if horizontal:
        ax.xaxis.set_ticks_position("bottom")
        ax.xaxis.set_label_position("top")
        cbar.ax.set_xlabel(label, size=size, labelpad=10)
        cbar.ax.tick_params(axis="x", labelsize=ticksize)
    else:
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")
        cbar.set_label(label, size=size, labelpad=18)
        cbar.ax.tick_params(axis="y", labelsize=ticksize)
    # ax.xaxis.set_ticks_position("top")
    # ax.xaxis.set_label_position("top")

    cbar.set_label(label, size=size, labelpad=18)

    cbar.ax.tick_params(labelsize=ticksize)
    # cbar.ax.set_xticks([0.5, 0.75, 1, 1.5, 2], labelsize=ticksize)
    return


def get_quantile(array, quantile):
    return np.percentile(array, quantile)


def get_histogram_2d(
    df,
    x_values,
    y_values,
    bins,
    color_prop="M_out",
    statistic="counts",
    quant=None,
):
    if statistic == "quantile":
        statistic = partial(get_quantile, quantile=quant)

    hist, *_ = stats.binned_statistic_2d(
        x_values,
        y_values,
        values=df[color_prop],
        statistic=statistic,
        bins=bins,
    )

    hist_cont, xedges_cont, yedges_cont, _ = stats.binned_statistic_2d(
        x_values,
        y_values,
        values=df[color_prop],
        statistic="count",
        bins=bins,
    )

    if statistic == "count":
        pass
        # hist = np.log10(hist)
    return hist, hist_cont, xedges_cont, yedges_cont


def prop_prop_histogram(
    df,
    prop_x,
    prop_y,
    color_prop="M_out",
    statistic="counts",
    em_weighted=False,
    log_x=False,
    log_y=False,
    bins_x=12,
    bins_y=12,
    color_log=False,
    contour=True,
    quantile=None,
    for_slides=False,
    title=None,
):
    label_colors(for_slides)
    if color_log:
        df["color_prop"] = np.log10(df[color_prop])
        color_prop = "color_prop"

    parameters = plot_parameters_comp()

    df.dropna(subset=prop_x, inplace=True)
    df.dropna(subset=prop_y, inplace=True)

    x_values = df[prop_x]
    y_values = df[prop_y]
    if log_x:
        x_values = np.ma.masked_invalid(np.log10(x_values))
    if log_y:
        y_values = np.ma.masked_invalid(np.log10(y_values))

    x_edges = np.linspace(x_values.min(), x_values.max(), bins_x)
    y_edges = np.linspace(y_values.min(), y_values.max(), bins_y)
    if "sfr_" in prop_y:
        y_edges = np.linspace(-3, 3, bins_y)

    if "v_" in prop_y:
        y_edges = np.linspace(0, 950, bins_y)

    hist, hist_cont, xedges_cont, yedges_cont = get_histogram_2d(
        df,
        x_values,
        y_values,
        bins=[x_edges, y_edges],
        color_prop=color_prop,
        statistic=statistic,
        quant=quantile,
    )
    old_hist = np.copy(hist)
    if statistic == "count":
        hist = np.log10(hist)

    cont_centers_x = (xedges_cont[1:] + xedges_cont[:-1]) / 2
    cont_centers_y = (yedges_cont[1:] + yedges_cont[:-1]) / 2
    x_grid, y_grid = np.meshgrid(x_edges, y_edges)
    # vmin, vcenter, vmax = get_color_limits(color_prop, statistic)
    # col_norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    if statistic == "count":
        col_norm = colors.TwoSlopeNorm(vmin=0, vcenter=1.5, vmax=3)
    else:
        # col_norm = colors.TwoSlopeNorm(vmin=6, vcenter=8, vmax=10)
        col_norm = colors.TwoSlopeNorm(vmin=6, vcenter=7, vmax=8)

    f, axs = plt.subplots(
        ncols=2,
        nrows=1,
        gridspec_kw={
            "hspace": 0.1,
            "wspace": 0.1,  # 0.3 * 0.75 * len(props_of_interest),
            "width_ratios": [24, 1],
            "height_ratios": [24],
        },
        figsize=[10, 8],
    )
    ax = axs[0]
    cax = axs[1]
    subfig = ax.pcolormesh(
        x_grid, y_grid, hist.T, norm=col_norm, cmap=plt.get_cmap("inferno")
    )
    jades = get_jades_data()
    if for_slides:
        color = "red"
    else:
        color = "blue"
    # ax.scatter(
    #     jades["M_star_log_Oiii"],
    #     jades["M_out_log_Oiii"],
    #     marker="*",
    #     s=150,
    #     color=color
    # )
    # ax.scatter(
    #     jades["M_star_log_Ha"],
    #     jades["M_out_log_Ha"],
    #     marker="*",
    #     s=150,
    #     color=color,
    #     label="JADES"
    # )
    # ax.scatter(
    #     jades["M_star_log_Oiii"],
    #     jades["v_out_Oiii"],
    #     c=jades["M_out_log_Oiii"],
    #     marker="*",
    #     s=150,
    #     # color='red',
    #     # norm=col_norm,
    #     # cmap=plt.get_cmap("inferno"),
    # )
    # ax.scatter(
    #     jades["M_star_log_Ha"],
    #     jades["v_out_Ha"],
    #     c=jades["M_out_log_Ha"],
    #     marker="*",
    #     s=150,
    #     # color='red',
    #     norm=col_norm,
    #     cmap=plt.get_cmap("inferno"),
    #     label="JADES"
    # )

    # ax.scatter(
    #     jades["M_star_log_Oiii"],
    #     jades["SFR_log_Oiii"],
    #     c=jades["M_out_log_Oiii"],
    #     marker="*",
    #     s=150,
    #     norm=col_norm,
    #     cmap=plt.get_cmap("inferno"),
    # )
    # ax.scatter(
    #     jades["M_star_log_Ha"],
    #     jades["SFR_log_Ha"],
    #     c=jades["M_out_log_Ha"],
    #     marker="*",
    #     s=150,
    #     norm=col_norm,
    #     cmap=plt.get_cmap("inferno"),
    #     label="JADES"
    # )
    ax.legend(fontsize=15, loc="upper left")

    for i in range(len(xedges_cont) - 1):
        for j in range(len(yedges_cont) - 1):
            label_base = old_hist[i, j]
            if statistic == "count":
                count = int(label_base)
            else:
                count = f"{label_base:.2f}"
            if label_base > 0:  # Only annotate non-zero bins
                ax.text(
                    xedges_cont[i] + (xedges_cont[i + 1] - xedges_cont[i]) / 2,
                    yedges_cont[j] + (yedges_cont[j + 1] - yedges_cont[j]) / 2,
                    count,
                    color="blue",
                    ha="center",
                    va="center",
                )
    # levels = get_levels(hist_cont, thresholds=[0.954, 0.683])
    # if contour:
    #     ax.contour(
    #         cont_centers_x,
    #         cont_centers_y,
    #         hist_cont.T,
    #         levels=levels,
    #         linewidths=4,
    #         linestyles=["dotted", "dashed", "solid"],
    #         colors="lightblue",
    #     )
    if statistic == "count":
        color_label = r"$\log(n_\mathrm{galaxies})$"
    else:
        if statistic == "quantile":
            color_label = rf"$M_{{\mathrm{{out,}} {quantile}}}[\log(M_\odot)]$"
            # color_label = prop_labels(color_prop) + f" {quantile} quantile"
        else:
            color_label = prop_labels(color_prop)
    create_color_bar_hist(
        f,
        cax,
        subfig=subfig,
        label=color_label,
        gap=True,
        horizontal=False,
    )
    ax.tick_params(labelsize=15)

    ax.set_xlabel(prop_labels(prop_x), size=25)
    ax.set_ylabel(prop_labels(prop_y), size=25)
    # ax.set_ylim(0, 300)
    if title is not None:
        ax.set_title(title, size=25)
    return


def plot_prop_maps_grouped(
    halo_id,
    df,
    snap,
    props,
    grid_size=100,
    method="GMM",
    group_props=None,
    dirs=[1, 2],
    sizebar_length=1,
    projection_angle_theta=None,
    projection_angle_phi=0,
    for_slides=False,
    serra=False,
):
    label_colors(for_slides)

    gridder = GasGridder(
        df=df,
        halo_id=halo_id,
        snap=snap,
        group_props=group_props,
        out_gas_sel=method,
        grid_size=grid_size,
        quants=props,
        projection_angle_theta=projection_angle_theta,
        projection_angle_phi=projection_angle_phi,
        serra=serra,
    )
    for prop in props:
        plot_prop_maps(gridder, prop, dirs, sizebar_length)
    return


def get_jades_data():
    data = {}
    data["M_star_log_Oiii"] = np.array([7.69, 7.60, 7.85, 8.09, 7.78, 8.63])
    data["M_star_log_Ha"] = np.array([8.54, 8.11, 7.73, 8.28, 7.81, 7.93, 7.85, 8.24])
    data["SFR_log_Oiii"] = np.array([0.09, 0.53, 0.61, 0.39, 0.41, 1.14])
    data["SFR_log_Ha"] = np.array([0.65, 0.74, 0.14, 0.34, 0.09, -0.67, 0.61, 0.01])
    data["M_out_log_Oiii"] = np.array([6.46, 7.07, 6.84, 6.56, 7.12, 8.26])
    data["M_out_log_Ha"] = np.array([6.74, 7.17, 6.00, 6.54, 6.03, 5.85, 6.67, 6.51])
    data["v_out_Oiii"] = np.array([500, 234, 701, 401, 259, 289])
    data["v_out_Ha"] = np.array([267, 444, 497, 229, 275, 648, 261, 911])
    return data


def prop_prop_scatter(
    df,
    prop_x,
    prop_y,
    color_prop="M_out",
    log_x=False,
    log_y=False,
    for_slides=False,
):
    label_colors(for_slides)
    parameters = plot_parameters_comp()

    df.dropna(subset=prop_x, inplace=True)
    df.dropna(subset=prop_y, inplace=True)

    x_values = df[prop_x]
    y_values = df[prop_y]
    if log_x:
        x_values = np.ma.masked_invalid(np.log10(x_values))
    if log_y:
        y_values = np.ma.masked_invalid(np.log10(y_values))

    col_norm = colors.TwoSlopeNorm(vmin=6, vcenter=8, vmax=10)
    data = get_jades_data()

    # f, axs = plt.subplots(
    #     ncols=2,
    #     nrows=1,
    #     gridspec_kw={
    #         "hspace": 0.1,
    #         "wspace": 0.1,  # 0.3 * 0.75 * len(props_of_interest),
    #         "width_ratios": [24, 1],
    #         "height_ratios": [24],
    #     },
    #     figsize=[10, 8],
    # )
    f, ax = plt.subplots(
        ncols=1,
        nrows=1,
        gridspec_kw={
            "hspace": 0.1,
            "wspace": 0.1,  # 0.3 * 0.75 * len(props_of_interest),
            "width_ratios": [24],
            "height_ratios": [24],
        },
        figsize=[10, 8],
    )
    # ax = axs[0]
    # cax = axs[1]
    if for_slides:
        color = "r"
    else:
        color = "b"
    subfig = ax.scatter(
        x_values, y_values, s=1, color=color, alpha=0.3, label="TNG data"
    )
    # ax.scatter(
    #     data["M_star_log_Oiii"],
    #     data["M_out_log_Oiii"],
    #     s=100,
    #     color="r",
    #     marker="*",
    #     label="JADES data OIII",
    # )
    # ax.scatter(
    #     data["M_star_log_Ha"],
    #     data["M_out_log_Ha"],
    #     s=100,
    #     color="yellow",
    #     marker="P",
    #     label=r"JADES data H$\alpha$",
    # )

    color_label = prop_labels(color_prop)
    # create_color_bar(
    #     f,
    #     cax,
    #     subfig=subfig,
    #     label=color_label,
    #     gap=True,
    #     horizontal=False,
    # )
    ax.tick_params(labelsize=15)
    # ax.legend(fontsize=15)

    ax.set_xlabel(prop_labels(prop_x), size=25)
    ax.set_ylabel(prop_labels(prop_y), size=25)
    return


def get_detection_fraction(df, thresholds, bins=20):
    df["bin"] = pd.cut(df["M_star_log"], bins=bins)
    fractions = []
    for threshold in thresholds:
        is_larger = (
            df.groupby("bin")["M_out_log"]
            .apply(np.array)
            .apply(lambda x: x > threshold)
        )
        fraction = []
        centers = []
        for bin in df["bin"].unique():
            centers.append((bin.left + bin.right) / 2)
        centers = np.sort(centers)

        for array in is_larger:
            fraction.append(np.sum(array) / len(array))
        fractions.append(fraction)

    f, ax = plt.subplots(
        ncols=1,
        nrows=1,
        gridspec_kw={
            "hspace": 0.1,
            "wspace": 0.1,  # 0.3 * 0.75 * len(props_of_interest),
            "width_ratios": [24],
            "height_ratios": [24],
        },
        figsize=[10, 8],
    )
    for i, element in enumerate(fractions):
        ax.plot(centers, element, label=rf"$10^{{{thresholds[i]}}} M_\odot$ threshold")
    ax.tick_params(labelsize=15)
    ax.legend(fontsize=15)

    ax.set_xlabel(prop_labels("M_out_log"), size=25)
    ax.set_ylabel("Fraction of detectable outflows", size=25)
    return


def plot_W80_evolution(
    df,
    theta_angles=[0, 30, 60, 90],
    phi_angles=[0],
    bins=100,
    cumulative=False,
    for_slides=False,
    title=None,
    aperture=False,
):
    label_colors(for_slides)
    bins = np.linspace(0, 5, 100)
    centers = (bins[1:] + bins[:-1]) / 2
    width = centers[1] - centers[0]
    fig, ax = plt.subplots(
        ncols=1,
        nrows=1,
        gridspec_kw={
            "hspace": 0.1,
            "wspace": 0.1,  # 0.3 * 0.75 * len(props_of_interest),
            "width_ratios": [24],
            "height_ratios": [24],
        },
        figsize=(10, 8),
    )
    for theta in theta_angles:
        for phi in phi_angles:
            if aperture:
                key1 = f"W80_outflow_{phi}_{theta}_aperture"
                key2 = f"W80_galaxy_{phi}_{theta}_aperture"
            else:
                key1 = f"W80_outflow_{phi}_{theta}"
                key2 = f"W80_galaxy_{phi}_{theta}"
            print(key1)
            ratios = df[key1] / df[key2]
            hist, _ = np.histogram(ratios, bins=bins, density=True)
            if cumulative:
                y_values = np.cumsum(hist * width)
                ax.plot(centers, y_values, label=f"{theta} degrees")
            else:
                y_values = hist
                ax.bar(
                    centers,
                    y_values,
                    width=centers[1] - centers[0],
                    alpha=0.5,
                    label=f"{theta} degrees",
                )
            if for_slides:
                color = "white"
            else:
                color = "black"
            ax.axvline(x=1.2, linestyle="--", color=color)
            ax.text(
                1,
                0.6,
                r"$W_{80,out}/W_{80,gal} = 1.2$",
                fontsize=12,
                rotation=90,
                color=color,
            )
            if title is not None:
                ax.set_title(title, fontsize=25)
    if cumulative:
        y_label = r"CDF($W_{80,out}/W_{80,gal}$)"
        ax.set_ylim(0, 1)
    else:
        y_label = r"PDF($W_{80,out}/W_{80,gal}$)"
    ax.tick_params(labelsize=15)
    ax.set_xlabel(r"$W_{80,out}/W_{80,gal}$", size=25)
    ax.set_ylabel(y_label, size=25)
    ax.legend(fontsize=15)
    return


def plot_galaxy_evolution(
    galaxies, prop_x, prop_y, color_prop, prop_y2=None, for_slides=False, title=None
):
    label_colors(for_slides)
    if color_prop is not None:
        f, axs = plt.subplots(
            ncols=1,
            nrows=2,
            gridspec_kw={
                "hspace": 0.2,
                "wspace": 0.1,  # 0.3 * 0.75 * len(props_of_interest),
                "width_ratios": [24],
                "height_ratios": [1, 24],
            },
            figsize=[10, 10],
        )
        col_norm = colors.TwoSlopeNorm(vmin=-9, vcenter=-8.25, vmax=-7.5)
        cax = axs[0]  # Positioning the color bar at the top
        ax = axs[1]
    else:
        f, ax = plt.subplots(figsize=[10, 8])
        cax = None
    col_norm = colors.TwoSlopeNorm(vmin=-9, vcenter=-8.25, vmax=-7.5)

    ax_right = ax.twinx() if prop_y2 else None

    for id, galaxy in galaxies.items():
        ax.plot(galaxy[prop_x], galaxy[prop_y], linewidth=2, color='red', label=f"$M_\star=10^{{{galaxy['M_star_log'][0]:.1f}}}M_\odot$")
        if color_prop is not None:
            subfig = ax.scatter(
                galaxy[prop_x],
                galaxy[prop_y],
                marker="*",
                c=galaxy[color_prop],
                cmap=plt.get_cmap("inferno"),
                s=500,
                norm=col_norm,
            )

        if prop_y2 and ax_right:
            ax_right.plot(
                galaxy[prop_x], galaxy[prop_y2], linewidth=2, linestyle="dashed", color="blue")
            # ax_right.scatter(
            #     galaxy[prop_x], galaxy[prop_y2],
            #     marker="o", color="blue", s=100,
            # )
        
    if color_prop is not None:
        color_label = prop_labels(color_prop)
        create_color_bar_hist(
            f,
            cax,
            subfig=subfig,
            label=color_label,
            gap=True,
            horizontal=True,
        )

    if ax_right:
        ax_right.set_ylabel(prop_labels(prop_y2), size=25, color="blue")
        ax_right.tick_params(axis='y', labelcolor="blue", labelsize=15)
        # ax_right.legend(fontsize=15, loc="upper right")
    
    ax.tick_params(labelsize=15)
    ax.tick_params(axis="y", labelsize=15, labelcolor="red")
    ax.set_xlabel(prop_labels(prop_x), size=25)
    ax.set_ylabel(prop_labels(prop_y), size=25, color="red")
    ax.legend(fontsize=15)

    return
