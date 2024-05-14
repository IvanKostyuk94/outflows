from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import colormaps
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

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
        parameters["vmin"] = 6.0
        parameters["vcenter"] = 7.5
        parameters["vmax"] = 9

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


def plot_parameters_comp(prop):
    parameters = {}

    parameters["titlesize"] = 30

    parameters["colorbar_labelsize"] = 25
    parameters["colorbar_ticklabelsize"] = 15

    parameters["height_per_image"] = 6
    parameters["width_per_image"] = 6

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


def plot_prop_maps_grouped(
    halo_id,
    df,
    snap,
    props,
    grid_size=100,
    group_props=None,
    n_peak=None,
    dirs=[1, 2],
    sizebar_length=1,
):

    gal = Galaxy(
        df=df,
        halo_id=halo_id,
        snap=snap,
        group_props=group_props,
        n_peak=n_peak,
    )
    gridder = GasGridder(
        gal=gal, grid_size=grid_size, grouped_selection=True, quants=props
    )
    for prop in props:
        plot_prop_maps(gridder, prop, dirs, sizebar_length)
    return
