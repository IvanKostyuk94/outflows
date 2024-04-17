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

from Grid_halo import grid_gas, retrieve_halo_gas
from gaussian_outflow_selection import group_gas


def prop_labels(prop):
    prop_labels = {
        "Flow_Velocities": r"$v_\mathrm{out}$",
        "Masses": r"$\Sigma[\log(M_\odot)\mathrm{kpc}^{-2}]$",
        "StarFormationRate": r"$\Sigma_\mathrm{SFR}[\log(M_\odot)\mathrm{yr}^{-1}\mathrm{kpc}^{-2}]$",
        "Temperature": r"$T[\log(K)]$",
        "GFM_Metallicity": r"$\log(Z)$",
    }
    return prop_labels[prop]


def plot_parameters_comp(prop):
    parameters = {}

    parameters["titlesize"] = 30

    parameters["colorbar_labelsize"] = 30
    parameters["colorbar_ticklabelsize"] = 20

    parameters["height_per_image"] = 6
    parameters["width_per_image"] = 6

    # if prop == "Flow_Velocities":
    #     parameters["vmin"] = 50
    #     parameters["vcenter"] = 150
    #     parameters["vmax"] = 250
    if prop == "Flow_Velocities":
        parameters["vmin"] = -250
        parameters["vcenter"] = 0
        parameters["vmax"] = 250

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

    return parameters


def get_pixel_length_abs(box_size, grid_shape, snap):
    z = get_redshift(snap)
    a = scale_factor(z)
    pixel_length_com = box_size / grid_shape[0]
    pixel_length_abs = pixel_length_com / TNGcosmo.h * a
    return pixel_length_abs


def get_surface_densities(gas, box_size, snap, ax):
    gas = gas["Masses"]
    cell_size = get_pixel_length_abs(box_size, gas.shape, snap)
    tot_mass_ax = gas.sum(axis=ax) * 1e10 / TNGcosmo.h
    surface_dens = np.log10(tot_mass_ax / cell_size**2 + 1e-9)
    return surface_dens


def get_sfr_densities(gas, box_size, snap, ax):
    sfrs = gas["StarFormationRate"]
    cell_size = get_pixel_length_abs(box_size, sfrs.shape, snap)
    tot_mass_ax = sfrs.sum(axis=ax)
    surface_dens = np.log10(tot_mass_ax / cell_size**2 + 1e-9)
    return surface_dens


def get_outflow_image(gas, axis):
    data = gas["Flow_Velocities"]
    image = np.where(
        (data != 0).sum(axis) != 0,
        np.true_divide(data.sum(axis), (data != 0).sum(axis)),
        0,
    )
    return image


def get_mass_weighted_image(gas, axis, prop, log=False):
    data = gas[prop] * gas["Masses"]
    masses = gas["Masses"]
    image = np.where(
        (data != 0).sum(axis) != 0,
        np.true_divide(data.sum(axis), masses.sum(axis)),
        0,
    )
    if log:
        image = np.log10(image + 1e-20)
    return image


def get_prop_image(
    gas,
    prop,
    axis,
    snap=None,
    box_size=None,
    general_image=False,
):
    if prop == "Flow_Velocities":
        if general_image:
            image = get_mass_weighted_image(gas, axis, prop="Flow_Velocities")
        else:
            image = get_outflow_image(gas, axis)
    elif prop == "GFM_Metallicity":
        image = get_mass_weighted_image(
            gas, axis, prop="GFM_Metallicity", log=True
        )
    elif prop == "Temperature":
        image = get_mass_weighted_image(
            gas, axis, prop="Temperature", log=True
        )
    elif prop == "Masses":
        image = get_surface_densities(gas, box_size, snap, axis)
    elif prop == "StarFormationRate":
        image = get_sfr_densities(gas, box_size, snap, axis)
    return image


def draw_sizebar(ax, box_size, grid_shape, snap, length_kpc=1):
    pixel_length_abs = get_pixel_length_abs(box_size, grid_shape, snap)
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
        # cbar.ax.set_yticks([100, 150, 200, 250], labelsize=ticksize)
        cbar.ax.set_yticks([-200, -100, 0, 100, 200], labelsize=ticksize)

    return


def plot_outflow_comparison(
    gas,
    out_gas,
    r_vir,
    prop,
    snap,
    sizebar_length,
    box_size,
    with_circle,
    threshold_vel=100,
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
                    box_size=box_size,
                )
                subfig = ax.pcolormesh(
                    image,
                    cmap=colormaps["inferno"],
                    norm=col_norm,
                )
                if with_circle:
                    draw_r_vir_circle(ax, r_vir, image.shape)
                draw_sizebar(
                    ax, r_vir, image.shape, snap, length_kpc=sizebar_length
                )
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


def plot_prop_maps(
    gas, r_vir, prop, snap, with_circle, box_size, sizebar_length
):
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
            "wspace": 0.05,
            "hspace": 0.07,
            "width_ratios": [12, 12, 12, 1],
        },
        figsize=figsize,
    )

    for column in range(4):
        ax = axs[column]
        if prop == "Flow_Velocities":
            cmap = "coolwarm"
        else:
            cmap = "inferno"
        if column < 3:
            image = get_prop_image(
                gas,
                prop,
                snap=snap,
                axis=column,
                box_size=box_size,
                general_image=True,
            )
            subfig = ax.pcolormesh(
                image,
                cmap=colormaps[cmap],
                norm=col_norm,
            )
            if with_circle:
                draw_r_vir_circle(ax, r_vir, image.shape)
            draw_sizebar(
                ax, box_size, image.shape, snap, length_kpc=sizebar_length
            )
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
    zoom_in=1,
    out_only=False,
    angle=None,
    v_out_threshold=None,
    v_esc_ratio=None,
):

    gas = grid_gas(
        halo_id,
        df,
        snap,
        out_only=out_only,
        threshold_velocity=v_out_threshold,
        v_esc_ratio=v_esc_ratio,
        grid_size=grid_size,
        zoom_in=zoom_in,
        projection_angle=angle,
    )

    r_vir = float(df[df.Halo_id == halo_id].R_vir)
    if zoom_in != 1:
        with_circle = False
        box_size = r_vir * 2 * float(config["cutout_scale"]) / zoom_in
        sizebar_length = 1
    else:
        with_circle = True
        box_size = r_vir * 2 / zoom_in
        sizebar_length = 10
    plot_prop_maps(
        gas, r_vir, prop, snap, with_circle, box_size, sizebar_length
    )
    return


def plot_prop_maps_grouped(
    halo_id,
    df,
    snap,
    props,
    grid_size=100,
    zoom_in=1,
    angle=None,
    group_props=["Flow_Velocities"],
    n_peak=None,
):

    gases = grid_gas(
        halo_id,
        df,
        snap,
        out_only=False,
        threshold_velocity=None,
        v_esc_ratio=None,
        grid_size=grid_size,
        zoom_in=zoom_in,
        projection_angle=angle,
        grouped_selection=True,
        group_props=group_props,
        n_peak=n_peak,
    )

    r_vir = float(df[df.Halo_id == halo_id].R_vir)
    if zoom_in != 1:
        with_circle = False
        box_size = r_vir * 2 * float(config["cutout_scale"]) / zoom_in
        sizebar_length = 1
    else:
        with_circle = True
        box_size = r_vir * 2 / zoom_in
        sizebar_length = 10
    for prop in props:
        for gas in gases:
            plot_prop_maps(
                gas, r_vir, prop, snap, with_circle, box_size, sizebar_length
            )
    return


def plot_pre_post_grid_comparison(
    halo_id,
    df,
    snap,
    prop,
    threshold_velocity=100,
    grid_size=100,
    zoom_in=1,
):
    gas = grid_gas(
        halo_id,
        df,
        snap,
        out_only=False,
        threshold_velocity=threshold_velocity,
        grid_size=grid_size,
        zoom_in=zoom_in,
    )
    out_gas = grid_gas(
        halo_id,
        df,
        snap,
        out_only=True,
        threshold_velocity=threshold_velocity,
        grid_size=grid_size,
        zoom_in=zoom_in,
    )

    r_vir = float(df[df.Halo_id == halo_id].R_vir)
    if zoom_in != 1:
        with_circle = False
        box_size = r_vir * 2 * float(config["cutout_scale"]) / zoom_in
        sizebar_length = 1
    else:
        with_circle = True
        box_size = r_vir * 2 / zoom_in
        sizebar_length = 10

    r_vir = float(df[df.Halo_id == halo_id].R_vir)
    plot_outflow_comparison(
        gas,
        out_gas,
        r_vir,
        prop,
        snap,
        with_circle=with_circle,
        box_size=box_size,
        sizebar_length=sizebar_length,
        threshold_vel=threshold_velocity,
    )
    return


def parameters_histogram():
    parameters = {}

    parameters["titlesize"] = 30

    parameters["fig_height"] = 10
    parameters["fig_width"] = 15

    parameters["axes_width"] = 3

    parameters["tick_major_size"] = 16
    parameters["tick_major_width"] = 4
    parameters["tick_minor_size"] = 8
    parameters["tick_minor_width"] = 3
    parameters["tick_labelsize"] = 20

    parameters["labelsize"] = 35
    parameters["label_x"] = r"$v_\mathrm{out}$"
    parameters["label_y"] = r"$P(v_\mathrm{out})$"
    parameters["alpha"] = 1
    parameters["legendsize"] = 10
    parameters["range"] = [-500, 500]
    # parameters["title"] = r"$M_\star>10^{10}M_\odot$ at $z=3$"
    # parameters["title"] = r"$10^{8}M_\odot<M_\star<10^{9}M_\odot$ at $z=3$"
    parameters["title"] = r"$10^{7}M_\odot<M_\star<10^{8}M_\odot$ at $z=3$"

    parameters["titlesize"] = 40

    return parameters


def get_outflow_velocities(df, snap, idces, grouped=False, peaks=None):
    data = {}
    data["outflow_velocities"] = np.array([])
    data["masses"] = np.array([])
    for id in idces:
        gas = retrieve_halo_gas(df, snap, id)
        data["outflow_velocities"] = np.append(
            data["outflow_velocities"], gas["Flow_Velocities"]
        )
        data["masses"] = np.append(data["masses"], gas["Masses"])
        if grouped:
            if len(idces) > 1:
                raise NotImplementedError(
                    "Cannot created grouped histogram for several galaxies"
                )
            else:
                _ = group_gas(
                    gas, props=["Flow_Velocities"], peak_number=peaks
                )
        data["group"] = gas["group"]
    return data


def plot_outflow_histogram(
    df, idces, snap, bins=100, grouped=False, peaks=None
):
    parameters = parameters_histogram()
    data = get_outflow_velocities(
        df, snap, idces, grouped=grouped, peaks=peaks
    )

    figsize = (parameters["fig_width"], parameters["fig_height"])

    _, ax = plt.subplots(figsize=figsize)
    ax.tick_params(
        length=parameters["tick_major_size"],
        width=parameters["tick_major_width"],
    )
    ax.tick_params(
        length=parameters["tick_minor_size"],
        width=parameters["tick_minor_width"],
        which="minor",
    )
    # ax.set_title(parameters["title"], size=parameters["titlesize"])
    if grouped:
        for i in range(np.max(data["group"] + 1)):
            norm = np.sum(data["group"] == i) / len(data["group"])
            height, bins = np.histogram(
                data["outflow_velocities"][data["group"] == i],
                bins=bins,
                density=True,
                weights=data["masses"][data["group"] == i],
                # alpha=parameters["alpha"],
                range=parameters["range"],
                # log=False,
            )
            bincentres = [
                (bins[i] + bins[i + 1]) / 2.0 for i in range(len(bins) - 1)
            ]
            plt.bar(bincentres, height * norm, width=bins[1:] - bins[:-1])

    else:
        ax.hist(
            data["outflow_velocities"],
            bins=bins,
            density=True,
            weights=data["masses"],
            alpha=parameters["alpha"],
            range=parameters["range"],
            log=False,
        )

    ax.set_xlabel(parameters["label_x"], size=parameters["labelsize"])
    ax.set_ylabel(parameters["label_y"], size=parameters["labelsize"])
    ax.tick_params(axis="both", labelsize=parameters["tick_labelsize"])
    plt.legend(fontsize=parameters["legendsize"])
    plt.show()
    return
