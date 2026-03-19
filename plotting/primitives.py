from matplotlib import pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from .config import plot_parameters_comp


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
        cbar.ax.set_yticks([-100, -50, 0, 50, 100, 100])
        cbar.ax.set_yticklabels(
            [-100, -50, 0, 50, 100, 100], fontsize=ticksize
        )
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
    return
