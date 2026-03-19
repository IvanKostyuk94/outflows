import numpy as np
import scipy
from matplotlib import pyplot as plt

from .config import plot_parameters_comp, label_colors, prop_labels
from los_projection import GalaxyProjections


def get_weights(gas, weighting=None):
    if weighting == "Luminosity":
        weights = (gas["Density"] * gas["Masses"]) / gas["Density"].mean()
    elif weighting == "Luminosity_light":
        test = gas["Density"] / gas["SFR_dist"] ** 2
        weights = (
            gas["Density"] * gas["Masses"] / gas["SFR_dist"] ** 2 / test.mean()
        )
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
        weights=weights,
    )
    return heights


def Gauss(x, a, x0, sigma):
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2))


def Gauss2(x, a, x0_1, sigma, delta_a, x0_2, delta_sigma):
    return (a * np.exp(-((x - x0_1) ** 2) / (2 * sigma**2))) + (
        a
        * delta_a
        * np.exp(-((x - x0_2) ** 2) / (2 * (sigma * delta_sigma) ** 2))
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
        bins = np.linspace(-1200, 1200, 80)
    else:
        bins = np.linspace(range[0], range[1], bin_n + 1)
    centers = (bins[1:] + bins[:-1]) / 2
    widths = bins[1:] - bins[:-1]
    tot_heights = np.zeros_like(centers)
    heights = None
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
    if heights is not None:
        bounds_1 = ([0, -np.inf, 0], [np.inf, np.inf, 150])
        popt, pcov = scipy.optimize.curve_fit(
            Gauss, centers, heights, p0=[max(heights), 0, 10], bounds=bounds_1
        )
        bounds_2 = (
            [0, -np.inf, 0, 0, -np.inf, 1.2],
            [np.inf, np.inf, 150, 0.5, np.inf, np.inf],
        )
        popt2, pcov2 = scipy.optimize.curve_fit(
            Gauss2,
            centers,
            heights,
            p0=[max(heights), 0, 30, 0.3, 0, 2],
            bounds=bounds_2,
        )
        pop_gauss1 = popt2[:3]
        pop_gauss2 = [popt2[0] * popt2[3], popt2[4], popt2[5] * popt2[2]]
        ax.plot(centers, Gauss(centers, *popt), "r-", label="fit Gauss")
        ax.plot(centers, Gauss(centers, *pop_gauss1), "b-", label="fit 2 Gauss 1")
        ax.plot(centers, Gauss(centers, *pop_gauss2), "g-", label="fit 2 Gauss 2")

    parameters = plot_parameters_comp()
    ax.set_xlabel(
        prop_labels(velocity_types[-1]), fontsize=parameters["label_fontsize"]
    )
    ax.set_ylabel(
        prop_labels(weighting[i]), fontsize=parameters["label_fontsize"]
    )
    ax.tick_params(labelsize=parameters["ticklabelsize"])
    ax.legend(fontsize=15)
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
    heights, _ = np.histogram(quantity, bins=bins)
    ax.bar(centers, heights)

    parameters = plot_parameters_comp()
    ax.set_xlabel(
        r"$n_e[\mathrm{cm}^{-3}]$", fontsize=parameters["label_fontsize"]
    )
    ax.set_ylabel(
        r"$\log(n_\mathrm{galaxies}$",
        fontsize=parameters["label_fontsize"],
    )
    ax.tick_params(labelsize=parameters["ticklabelsize"])
    ax.legend(fontsize=15)
    if title is not None:
        ax.set_title(title, fontsize=25)
    return


def plot_los_histograms(
    halo_id, snap, df, angles_theta, angles_phi, backend, bin_n=100
):
    from utils import get_halo

    columns = len(angles_theta)
    rows = len(angles_phi)
    figsize = (5 * columns, 5 * rows)
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
                backend=backend,
            )
            gal.project_outflows()

            gases = [gal.gas, gal.out_gas, gal.remain_gas]
            labels = ["all", "out", "remain"]
            velocity_types = ["los_Velocities", "los_Velocities", "los_Velocities"]

            bins = np.linspace(
                gases[0][velocity_types[0]].min(),
                gases[0][velocity_types[0]].max(),
                bin_n,
            )
            centers = (bins[1:] + bins[:-1]) / 2
            widths = bins[1:] - bins[:-1]
            for k, gas in enumerate(gases):
                weights = get_weights(gas)
                heights = get_histogram(gas, velocity_types[k], bins, weights)
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


def get_W80(gas):
    v_range = np.quantile(gas["los_Velocities"], [0.1, 0.9])
    W80 = v_range[1] - v_range[0]
    return W80


def add_w80(ax, quantiles, out=False):
    y_frac = 0.2 if out else 0.35
    y_frac_shift = 0.015
    y_min, y_max = ax.get_ylim()
    y = y_min + y_frac * (y_max - y_min)
    shift = y_frac_shift * (y_max - y_min)
    ax.hlines(
        y=y,
        xmin=quantiles[0],
        xmax=quantiles[1],
        color="black",
        linewidth=3,
    )
    ax.text(
        0.5 * (quantiles[0] + quantiles[1]),
        y=y + shift,
        s="W80 out" if out else "W80 remain",
        color="black",
        ha="center",
        va="bottom",
        fontsize=10,
    )
    return


def add_textbox(ax, quantile_out, quantile_remain):
    w80_out = quantile_out[1] - quantile_out[0]
    w80_remain = quantile_remain[1] - quantile_remain[0]
    ratio = w80_out / w80_remain
    ax.text(
        0.03,
        0.93,
        rf"$W_{{80,\mathrm{{out}}}} / W_{{80,\mathrm{{gal}}}} = {ratio:.2f}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="black",
            alpha=0.9,
        ),
    )


def w80_histogram_single(df, halo_id, snap, backend):
    galaxy = GalaxyProjections(
        df, halo_id=halo_id, snap=snap, projection_angle_theta=0, backend=backend
    )
    galaxy.project_outflows()
    remain_0 = galaxy.remain_gas["los_Velocities"].copy()
    remain_0_quant = np.percentile(remain_0, [10, 90])
    outflow_0 = galaxy.out_gas["los_Velocities"].copy()
    outflow_0_quant = np.percentile(outflow_0, [10, 90])

    galaxy = GalaxyProjections(
        df, halo_id=halo_id, snap=snap, projection_angle_theta=90, backend=backend
    )
    galaxy.project_outflows()
    remain_90 = galaxy.remain_gas["los_Velocities"].copy()
    remain_90_quant = np.percentile(remain_90, [10, 90])
    outflow_90 = galaxy.out_gas["los_Velocities"].copy()
    outflow_90_quant = np.percentile(outflow_90, [10, 90])

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 4),
        sharey=True,
        gridspec_kw={"wspace": 0.0},
    )

    bins = 50
    alpha = 0.6
    axes[0].hist(
        remain_0,
        bins=bins,
        density=True,
        alpha=alpha,
        color="green",
        label=r"$v_\mathrm{los, gal}$ at 0 degrees",
    )
    add_w80(axes[0], remain_0_quant)
    axes[0].hist(
        outflow_0,
        bins=bins,
        density=True,
        alpha=alpha,
        color="red",
        label=r"$v_\mathrm{los, out}$ at 0 degrees",
    )
    add_w80(axes[0], outflow_0_quant, out=True)
    add_textbox(axes[0], outflow_0_quant, remain_0_quant)

    axes[0].set_xlabel("Velocity [km/s]", fontsize=12)
    axes[0].set_ylabel("Normalized counts", fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].tick_params(axis="both", which="major", labelsize=10)

    axes[1].hist(
        remain_90,
        bins=bins,
        density=True,
        alpha=alpha,
        color="green",
        label=r"$v_\mathrm{los, gal}$ at 90 degrees",
    )
    add_w80(axes[1], remain_90_quant)
    axes[1].hist(
        outflow_90,
        bins=bins,
        density=True,
        alpha=alpha,
        color="red",
        label=r"$v_\mathrm{los, out}$ at 90 degrees",
    )
    add_w80(axes[1], outflow_90_quant, out=True)
    add_textbox(axes[1], outflow_90_quant, remain_90_quant)

    axes[1].set_xlabel("Velocity [km/s]", fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].tick_params(axis="both", which="major", labelsize=10)

    plt.tight_layout()
    plt.show()
    return


def plot_distributions(df, id, snap, backend):
    gal_0 = GalaxyProjections(
        df=df,
        halo_id=id,
        snap=snap,
        projection_angle_theta=0,
        aperture_size=0.6,
        backend=backend,
    )
    gal_0.project_outflows()
    gal_90 = GalaxyProjections(
        df=df,
        halo_id=id,
        snap=snap,
        projection_angle_theta=90,
        aperture_size=0.6,
        backend=backend,
    )
    gal_90.project_outflows()
    remain_gas_0 = gal_0.get_in_aperture(gal_0.remain_gas)
    out_gas_0 = gal_0.get_in_aperture(gal_0.out_gas)
    remain_gas_90 = gal_90.get_in_aperture(gal_90.remain_gas)
    out_gas_90 = gal_90.get_in_aperture(gal_90.out_gas)

    W80_galaxy_0 = get_W80(remain_gas_0)
    W80_outflow_0 = get_W80(out_gas_0)
    W80_galaxy_90 = get_W80(remain_gas_90)
    W80_outflow_90 = get_W80(out_gas_90)

    mstar = df[(df.idx == id) & (df.snap == snap)]["M_star_log"].values[0]
    title = (
        f"Galaxy {id} at z={snap}, Mstar=10^{mstar:.1f}Msun: "
        f"W80_ratio= {W80_outflow_0/W80_galaxy_0:.2f} (0), "
        f"{W80_outflow_90/W80_galaxy_90:.2f} (90)"
    )
    f, ax = plt.subplots(figsize=[10, 8])
    ax.set_title(title, size=20)
    ax.hist(
        gal_0.remain_gas["los_Velocities"],
        bins=100,
        weights=gal_0.remain_gas["mass"],
        alpha=0.5,
        label="remain_0",
    )
    ax.hist(
        gal_0.out_gas["los_Velocities"],
        bins=100,
        weights=gal_0.out_gas["mass"],
        alpha=0.5,
        label="out_0",
    )
    ax.hist(
        gal_90.remain_gas["los_Velocities"],
        bins=100,
        weights=gal_90.remain_gas["mass"],
        alpha=0.5,
        label="remain_90",
    )
    ax.hist(
        gal_90.out_gas["los_Velocities"],
        bins=100,
        weights=gal_90.out_gas["mass"],
        alpha=0.5,
        label="out_90",
    )
    ax.set_xlabel("v [km/s]", size=20)
    ax.set_ylabel("M [Msun]", size=20)
    ax.legend()
    return


def plot_mass_histograms(df_tng, df_serra, bins=30, for_slides=False):
    label_colors(for_slides)

    mass_key = "M_star_log"
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        df_tng[mass_key], bins=bins, alpha=0.5, label="TNG50", color="cyan"
    )
    ax.hist(
        df_serra[mass_key],
        bins=bins,
        alpha=0.5,
        label="SERRA",
        color="magenta",
    )

    ax.axvspan(7.6, 8.63, alpha=0.70, color="grey")

    ymin, ymax = ax.get_ylim()
    textcolor = "white" if for_slides else "black"
    ax.text(
        (7.6 + 8.63) / 2,
        ymax * 0.9,
        "JADES range",
        ha="center",
        va="center",
        fontsize=16,
        color=textcolor,
    )

    ax.set_xlabel(prop_labels(mass_key), fontsize=25)
    ax.set_ylabel("count", fontsize=25)
    ax.set_yscale("log")
    ax.tick_params(axis="both", labelsize=15)
    ax.legend(fontsize=15)
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
    import numpy as np
    from matplotlib import pyplot as plt

    label_colors(for_slides)
    bins = np.linspace(0, 5, 100)
    centers = (bins[1:] + bins[:-1]) / 2
    width = centers[1] - centers[0]
    fig, ax = plt.subplots(
        ncols=1,
        nrows=1,
        gridspec_kw={
            "hspace": 0.1,
            "wspace": 0.1,
            "width_ratios": [24],
            "height_ratios": [24],
        },
        figsize=(10, 8),
    )
    plot_colors = ["#00FFFF", "#FFA500", "#00FF00", "#FF00FF"]
    for i, theta in enumerate(theta_angles):
        for phi in phi_angles:
            if aperture:
                key1 = f"W80_outflow_{phi}_{theta}_aperture"
                key2 = f"W80_galaxy_{phi}_{theta}_aperture"
            else:
                key1 = f"W80_outflow_{phi}_{theta}"
                key2 = f"W80_galaxy_{phi}_{theta}"
            ratios = df[key1] / df[key2]
            hist, _ = np.histogram(ratios, bins=bins, density=True)
            if cumulative:
                y_values = 1 - np.cumsum(hist * width)
                ax.plot(
                    centers,
                    y_values,
                    color=plot_colors[i],
                    label=f"{theta} degrees",
                    linewidth=3,
                )
            else:
                y_values = hist
                ax.bar(
                    centers,
                    y_values,
                    width=centers[1] - centers[0],
                    alpha=0.5,
                    label=f"{theta} degrees",
                )
            line_color = "white" if for_slides else "black"
            ax.axvline(x=1.2, linestyle="--", color=line_color)
            ax.text(
                1,
                0.3,
                r"$W_{80,out}/W_{80,gal} = 1.2$",
                fontsize=12,
                rotation=90,
                color=line_color,
            )
            if title is not None:
                ax.set_title(title, fontsize=25)
    if cumulative:
        y_label = r"1-CDF($W_{80,out}/W_{80,gal}$)"
        ax.set_ylim(0, 1)
    else:
        y_label = r"PDF($W_{80,out}/W_{80,gal}$)"
    ax.tick_params(labelsize=15)
    ax.set_xlabel(r"$W_{80,out}/W_{80,gal}$", size=25)
    ax.set_ylabel(y_label, size=25)
    ax.legend(fontsize=15)
    plt.savefig(
        "orientation_serra.png",
        facecolor="black",
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0,
    )
    return
