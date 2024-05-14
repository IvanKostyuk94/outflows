# A bunch of old plotting functionality that might be usefull in the future


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
