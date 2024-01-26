import numpy as np
from matplotlib import pyplot as plt
from config import config
from utils import *
from pyTNG.cosmology import TNGcosmo
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_parameters_comp():
    parameters = {}
    parameters["labelsize"] = 50

    parameters["titlesize"] = 30
    parameters["multiple_titlesize"] = 35
    parameters["length_major_ticks"] = 16
    parameters["length_minor_ticks"] = 8
    parameters["width_minor_ticks"] = 3
    parameters["width_major_ticks"] = 4
    parameters["labelsize_ticks"] = 35

    parameters["colorbar_labelsize"] = 50
    parameters["colorbar_ticklabelsize"] = 30
    parameters["colorbar_labelsize_multiple"] = 40
    parameters["colorbar_ticklabelsize_multiple"] = 30

    parameters["axes_width"] = 3

    parameters["figure_width"] = 15
    parameters["figure_height"] = 15

    parameters["height_per_image"] = 6
    parameters["width_per_image"] = 6

    parameters["x_label_convergence"] = r"Grid cells"
    parameters["y_label_convergence"] = r"$f_\mathrm{esc}$"

    parameters["nx"] = 45
    parameters["ny"] = 30

    parameters["v_min"] = -4
    parameters["v_max"] = 0

    parameters["x_lim_min"] = -4.8
    parameters["x_lim_max"] = 0

    parameters["y_lim_min"] = 2.8
    parameters["y_lim_max"] = 6.4

    parameters["legendsize"] = 30

    parameters["linewidth"] = 3
    parameters["capsize"] = 10
    parameters["capwidth"] = 3
    return parameters


def draw_sizebar(ax, r_vir, grid_shape, snap, length_kpc=1):
    """
    Draw a horizontal bar with length of 0.1 in data coordinates,
    with a fixed label underneath.
    """
    z = get_redshift(snap)
    a = scale_factor(z)
    pixel_length_com = (
        2 * float(config["cutout_scale"]) * r_vir / grid_shape[0]
    )
    pixel_length_abs = pixel_length_com / TNGcosmo.h * a

    length_bar = length_kpc / pixel_length_abs
    asb = AnchoredSizeBar(
        ax.transData,
        length_bar,
        f"{length_kpc}kpc",
        size_vertical=0.5,
        loc="lower center",
        pad=0.5,
        borderpad=0.5,
        sep=5,
        frameon=False,
        color="white",
    )
    ax.add_artist(asb)

def create_color_bar(
    f,
    ax,
    parameters,
    subfig,
    label=None,
    ax_is_cbar=False,
    horizontal=False,
    gap=False,
):
    if ax_is_cbar:
        if horizontal:
            cbar = f.colorbar(subfig, cax=ax)
            # ax.xaxis.set_ticks_position("top")
            # ax.xaxis.set_label_position("top")

        else:
            cbar = f.colorbar(subfig, cax=ax, orientation="horizontal")
            ax.xaxis.set_ticks_position("top")
            ax.xaxis.set_label_position("top")

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
    # if horizontal:
    #     cbar.ax.set_yticks([0.5, 0.75, 1, 1.5, 2], labelsize=ticksize)
    # else:
    #     cbar.ax.set_xticks([0.5, 0.75, 1, 1.5, 2], labelsize=ticksize)

    return



def plot_outflow_comparison(gas, out_gas, r_vir):
    parameters = plot_parameters_comp()

    image_columns = 3
    image_rows = 2
    figsize = (
        parameters["width_per_image"] * image_columns,
        parameters["height_per_image"] * image_rows,
    )
    fig, axs = plt.subplots(
        ncols=image_columns + 1,
        nrows=image_rows,
        gridspec_kw={
            "hspace": 0.01,
            "wspace": 0.05,
            "width_ratios": [12, 12, 12, 1],
        },
        figsize=figsize,
    )
    create_color_bar(
        fig,
        ax_col,
        parameters,
        subfig,
        label=label,
        multiple=True,
        ax_is_cbar=True,
        horizontal=horizontal,
        prop=prop,
    )
    ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
