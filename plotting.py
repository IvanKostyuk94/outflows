import numpy as np
from matplotlib import pyplot as plt
from config import config
from utils import get_redshift, scale_factor
from pyTNG.cosmology import TNGcosmo
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colormaps
from matplotlib.patches import Circle

from Grid_halo import grid_gas


def prop_labels(prop):
    prop_labels = {
        "Flow_Velocities": r"$v_\mathrm{out}$",
        "Masses": r"$\Sigma[\log(M_\odot)\mathrm{kpc}^{-2}]$",
        "StarFormationRate": r"$\Sigma_\mathrm{SFR}[\log(M_\odot)\mathrm{yr}^{-1}\mathrm{kpc}^{-2}]$",
        "Temperature": "K",
    }
    return prop_labels[prop]


def plot_parameters_comp(prop):
    parameters = {}

    parameters["titlesize"] = 30

    parameters["colorbar_labelsize"] = 30
    parameters["colorbar_ticklabelsize"] = 20

    parameters["height_per_image"] = 6
    parameters["width_per_image"] = 6

    if prop == "Flow_Velocities":
        parameters["vmin"] = 50
        parameters["vcenter"] = 150
        parameters["vmax"] = 250
    elif prop == "Temperature":
        parameters["vmin"] = 5.5
        parameters["vcenter"] = 6
        parameters["vmax"] = 6.5

    elif prop == "Masses":
        parameters["vmin"] = 3
        parameters["vcenter"] = 6
        parameters["vmax"] = 9

    elif prop == "StarFormationRate":
        parameters["vmin"] = -6
        parameters["vcenter"] = -3
        parameters["vmax"] = 0

    return parameters


def get_pixel_length_abs(r_vir, grid_shape, snap):
    z = get_redshift(snap)
    a = scale_factor(z)
    pixel_length_com = (
        2 * float(config["cutout_scale"]) * r_vir / grid_shape[0]
    )
    pixel_length_abs = pixel_length_com / TNGcosmo.h * a
    return pixel_length_abs


def get_surface_densities(gas, r_vir, snap, ax):
    cell_size = get_pixel_length_abs(r_vir, gas.shape, snap)
    tot_mass_ax = gas.sum(axis=ax)
    surface_dens = np.log10(tot_mass_ax / cell_size**2 + 1e-9)
    return surface_dens


def get_sfr_densities(sfrs, r_vir, snap, ax):
    cell_size = get_pixel_length_abs(r_vir, sfrs.shape, snap)
    tot_mass_ax = sfrs.sum(axis=ax)
    surface_dens = np.log10(tot_mass_ax / cell_size**2 + 1e-9)
    return surface_dens


def get_outflow_image(gas, outflow_only, axis, threshold_vel):
    if outflow_only:
        data = gas["Flow_Velocities"]
    else:
        data = np.where(
            gas["Flow_Velocities"] > threshold_vel, gas["Flow_Velocities"], 0
        )

    image = np.where(
        (data != 0).sum(axis) != 0,
        np.true_divide(data.sum(axis), (data != 0).sum(axis)),
        0,
    )
    return image


def get_prop_image(
    gas,
    prop,
    axis,
    outflow_only=None,
    threshold_vel=None,
    r_vir=None,
    snap=None,
):
    if prop == "Flow_Velocities":
        image = get_outflow_image(gas, outflow_only, axis, threshold_vel)
    elif prop == "Masses":
        image = get_surface_densities(gas, r_vir, snap, axis)
    elif prop == "StarFormationRate":
        image = get_surface_densities(gas, r_vir, snap, axis)
    return image


def draw_sizebar(ax, r_vir, grid_shape, snap, length_kpc=1):
    pixel_length_abs = get_pixel_length_abs(r_vir, grid_shape, snap)
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


def draw_r_vir_circle(ax, r_vir, grid_shape):
    pixel_length_com = (
        2 * float(config["cutout_scale"]) * r_vir / grid_shape[0]
    )
    r_vir_pix = r_vir / pixel_length_com
    center = grid_shape[0] / 2
    circ = Circle(
        (center, center),
        r_vir_pix,
        facecolor="None",
        linestyle="--",
        edgecolor="w",
        lw=2,
    )
    ax.add_patch(circ)
    return


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
        cbar.ax.set_yticks([100, 150, 200, 250], labelsize=ticksize)

    return


def plot_outflow_comparison(
    gas, out_gas, r_vir, prop, snap, threshold_vel=100
):
    parameters = plot_parameters_comp(prop)

    image_columns = 4
    image_rows = 2
    figsize = (
        parameters["width_per_image"] * (image_columns - 11 / 12),
        parameters["height_per_image"] * image_rows,
    )
    col_norm = colors.TwoSlopeNorm(
        vmin=parameters["vmin"],
        vcenter=parameters["vcenter"],
        vmax=parameters["vmax"],
    )

    fig, axs = plt.subplots(
        ncols=image_columns,
        nrows=image_rows,
        gridspec_kw={
            "hspace": 0.15,
            "wspace": 0.1,
            "width_ratios": [12, 12, 12, 1],
        },
        figsize=figsize,
    )

    for column in range(4):
        for row in range(2):
            ax = axs[row, column]
            if column < 3:
                if row == 0:
                    outflow_only = True
                    data = out_gas
                else:
                    outflow_only = False
                    data = gas

                image = get_prop_image(
                    data,
                    prop,
                    outflow_only=outflow_only,
                    axis=column,
                    threshold_vel=threshold_vel,
                )
                subfig = ax.pcolormesh(
                    image,
                    cmap=colormaps["inferno"],
                    norm=col_norm,
                )

                draw_r_vir_circle(ax, r_vir, image.shape)
                draw_sizebar(ax, r_vir, image.shape, snap, length_kpc=10)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                if column == 1:
                    if row == 0:
                        label = r"$v_\mathrm{out}$ selection before gridding"
                    if row == 1:
                        label = r"$v_\mathrm{out}$ selection after gridding"
                    ax.set_title(label, fontsize=parameters["titlesize"])
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


def plot_prop_maps(gas, r_vir, prop, snap):
    parameters = plot_parameters_comp(prop)

    image_columns = 4
    image_rows = 1
    figsize = (
        parameters["width_per_image"] * (image_columns - 11 / 12),
        parameters["height_per_image"] * image_rows,
    )
    col_norm = colors.TwoSlopeNorm(
        vmin=parameters["vmin"],
        vcenter=parameters["vcenter"],
        vmax=parameters["vmax"],
    )

    fig, axs = plt.subplots(
        ncols=image_columns,
        nrows=image_rows,
        gridspec_kw={
            "hspace": 0.15,
            "width_ratios": [12, 12, 12, 1],
        },
        figsize=figsize,
    )

    for column in range(4):
        ax = axs[column]
        if column < 3:
            data = gas[prop]
            image = get_prop_image(
                data,
                prop,
                r_vir=r_vir,
                snap=snap,
                axis=column,
            )
            subfig = ax.pcolormesh(
                image,
                cmap=colormaps["inferno"],
                norm=col_norm,
            )

            draw_r_vir_circle(ax, r_vir, image.shape)
            # draw_sizebar(ax, r_vir, image.shape, snap, length_kpc=10)
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


def retrieve_prop_maps(
    halo_id,
    df,
    snap,
    prop,
    grid_size=100,
):
    gas = grid_gas(
        halo_id,
        df,
        snap,
        out_only=False,
        grid_size=grid_size,
    )
    r_vir = float(df[df.Halo_id == halo_id].R_vir)
    plot_prop_maps(gas, r_vir, prop, snap)
    return


def plot_pre_post_grid_comparison(
    halo_id,
    df,
    snap,
    prop,
    threshold_velocity=100,
    grid_size=100,
):
    gas = grid_gas(
        halo_id,
        df,
        snap,
        out_only=False,
        threshold_velocity=threshold_velocity,
        grid_size=grid_size,
    )
    out_gas = grid_gas(
        halo_id,
        df,
        snap,
        out_only=True,
        threshold_velocity=threshold_velocity,
        grid_size=grid_size,
    )

    r_vir = float(df[df.Halo_id == halo_id].R_vir)
    plot_outflow_comparison(
        gas, out_gas, r_vir, prop, snap, threshold_vel=threshold_velocity
    )
    return
