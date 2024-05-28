from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import colormaps
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from utils import get_redshift
from Grid_halo import GasGridder
from process_gas import Galaxy


def prop_labels(prop):
    prop_labels = {
        "Flow_Velocities": r"$v_\mathrm{out}$",
        "Masses": r"$\Sigma[\log(M_\odot)\mathrm{kpc}^{-2}]$",
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
    }
    return prop_labels[prop]


def get_ranges(prop, parameters):
    # if prop == "Flow_Velocities":
    #     parameters["vmin"] = 50
    #     parameters["vcenter"] = 150
    #     parameters["vmax"] = 250
    if prop == "Flow_Velocities":
        parameters["vmin"] = -250
        parameters["vcenter"] = 0
        parameters["vmax"] = 250

    if prop == "Rot_Velocities":
        parameters["vmin"] = 0
        parameters["vcenter"] = 1500
        parameters["vmax"] = 3000

    if prop == "Angular_Velocities":
        parameters["vmin"] = 0
        parameters["vcenter"] = 300
        parameters["vmax"] = 600

    elif prop == "Masses":
        parameters["vmin"] = 7.0
        parameters["vcenter"] = 8.5
        parameters["vmax"] = 10

    # elif prop == "Masses":
    #     parameters["vmin"] = 4.0
    #     parameters["vcenter"] = 6.0
    #     parameters["vmax"] = 8.0

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
    }
    if prop in coolwarm_props:
        cmap = "coolwarm"
    else:
        cmap = "inferno"
    return colormaps[cmap]


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
        cbar.ax.set_yticks([-200, -100, 0, 100, 200], labelsize=ticksize)
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

    for row in range(rows):
        for column in range(columns):
            ax = axs[row, column]
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
                    label=prop_labels(prop),
                    ax_is_cbar=True,
                    horizontal=False,
                    prop=prop,
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


def plot_prop_correlation(df, prop_x, prop_y, bins=20, by_z=False, stepsize=1):
    x_centers, y_means_all, y_errors_all, labels = get_data(
        df=df, prop_x=prop_x, prop_y=prop_y, bins=bins, by_z=by_z
    )
    fig, ax = plt.subplots(figsize=(15, 10))
    if not by_z:
        ax.scatter(df[prop_x], df[prop_y], s=1, alpha=0.3, color="red")
    for i in range(0, len(y_means_all), stepsize):
        # ax.errorbar(
        #     x_centers,
        #     y_means_all[i],
        #     yerr=y_errors_all[i],
        #     fmt="o-",
        #     capsize=5,
        #     linewidth=3,
        #     label=labels[i],
        # )
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
    ax.set_ylim(0, 300)
    ax.set_xlim(7.5, 11)
    ax.legend(fontsize=15)

    return


def plot_prop_maps_grouped(
    halo_id,
    df,
    snap,
    props,
    grid_size=100,
    method="GMM",
    group_props=None,
    n_peak=3,
    dirs=[1, 2],
    sizebar_length=1,
):

    gal = Galaxy(
        df=df,
        halo_id=halo_id,
        snap=snap,
        group_props=group_props,
        n_peak=n_peak,
        out_gas_sel=method,
        with_rotation=True,
    )
    gridder = GasGridder(gal=gal, grid_size=grid_size, quants=props)
    for prop in props:
        plot_prop_maps(gridder, prop, dirs, sizebar_length)
    return
