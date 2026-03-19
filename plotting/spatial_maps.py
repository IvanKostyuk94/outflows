from matplotlib import pyplot as plt

from .config import plot_parameters_comp, get_cmap, label_colors, prop_labels
from .primitives import setup_prop_parameters, get_col_norm, draw_sizebar, create_color_bar
from Grid_halo import GasGridder


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
                color = "white"
                ax.text(6, 180, gas_types[row], fontsize=20, color=color)
                if (row == 0) and (prop == "Masses"):
                    ax.text(
                        120,
                        180,
                        r"$M_\star = 10^{8}\mathrm{M}_\odot$",
                        fontsize=20,
                        color=color,
                    )
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


def plot_prop_maps_grouped(
    halo_id,
    df,
    snap,
    props,
    backend,
    grid_size=100,
    method="GMM",
    group_props=None,
    dirs=[1, 2],
    sizebar_length=1,
    projection_angle_theta=None,
    projection_angle_phi=0,
    for_slides=False,
):
    label_colors(for_slides)

    gridder = GasGridder(
        df=df,
        halo_id=halo_id,
        snap=snap,
        backend=backend,
        group_props=group_props,
        out_gas_sel=method,
        grid_size=grid_size,
        quants=props,
        projection_angle_theta=projection_angle_theta,
        projection_angle_phi=projection_angle_phi,
    )
    for prop in props:
        plot_prop_maps(gridder, prop, dirs, sizebar_length)
    plt.savefig(
        f"{halo_id}_{snap}.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="black",
        edgecolor="none",
        pad_inches=0,
    )
    return
