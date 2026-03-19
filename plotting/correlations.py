import numpy as np
import pandas as pd
import pickle
from functools import partial
from matplotlib import pyplot as plt
from matplotlib import colors
import matplotlib.lines as mlines
from matplotlib.patches import Patch
from scipy import stats
from scipy.stats import gaussian_kde

from .config import plot_parameters_comp, label_colors, prop_labels, get_universe_age
from .primitives import create_color_bar_hist
from .observational import get_jades_data
from utils import get_redshift


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


def get_quantile(array, quantile):
    return np.percentile(array, quantile)


def get_histogram_2d(
    df,
    x_values,
    y_values,
    bins,
    bins_cont=None,
    color_prop="M_out",
    statistic="counts",
    quant=None,
):
    if bins_cont is None:
        bins_cont = bins
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
        bins=bins_cont,
    )
    return hist, hist_cont, xedges_cont, yedges_cont


def get_levels(hist_cont, thresholds):
    levels = []
    counts = np.sort(hist_cont.flatten())[::-1]
    value_thresholds = counts.sum() * np.array(thresholds)
    for threshold in value_thresholds:
        count_sum = 0
        i = 0
        while count_sum < threshold:
            count_sum += counts[i]
            i += 1
        levels.append(counts[i])
    return levels


def find_kde_level(cumsum, Z_sorted, fraction):
    return Z_sorted[np.searchsorted(cumsum, fraction)]


def get_kde_histogram(x_values, y_values, serra=False):
    xmin, xmax = x_values.min(), x_values.max()
    ymin, ymax = y_values.min(), y_values.max()
    data = np.vstack([x_values, y_values])
    kde = gaussian_kde(data)
    if serra:
        X, Y = np.meshgrid(
            np.linspace(xmin * 0.99, xmax * 1.1, 200),
            np.linspace(ymin, ymax * 1.1, 200),
        )
    else:
        X, Y = np.meshgrid(
            np.linspace(xmin, xmax, 200), np.linspace(ymin, ymax, 200)
        )
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = kde(positions).reshape(X.shape)

    Z_flat = Z.flatten()
    Z_sorted = np.sort(Z_flat)[::-1]
    cumsum = np.cumsum(Z_sorted)
    cumsum /= cumsum[-1]

    if serra:
        min_level = find_kde_level(cumsum, Z_sorted, 0)
        level_68 = find_kde_level(cumsum, Z_sorted, 0.68)
        level_95 = find_kde_level(cumsum, Z_sorted, 0.95)
        level_99 = find_kde_level(cumsum, Z_sorted, 0.997)
        levels = sorted([level_99, level_95, level_68, min_level])
    else:
        level_68 = find_kde_level(cumsum, Z_sorted, 0.68)
        level_95 = find_kde_level(cumsum, Z_sorted, 0.95)
        level_99 = find_kde_level(cumsum, Z_sorted, 0.997)
        levels = sorted([level_99, level_95, level_68])

    return X, Y, Z, levels


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
    with_contours=False,
    with_fits=False,
):
    label_colors(for_slides)
    if color_log:
        df["color_prop"] = np.log10(df[color_prop])
        color_prop = "color_prop"

    df = df.copy(deep=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=[prop_x, prop_y], inplace=True)

    x_values = df[prop_x]
    y_values = df[prop_y]
    if log_x:
        x_values = np.ma.masked_invalid(np.log10(x_values))
    if log_y:
        y_values = np.ma.masked_invalid(np.log10(y_values))

    x_edges = np.linspace(x_values.min(), x_values.max(), bins_x)
    y_edges = np.linspace(y_values.min(), y_values.max(), bins_y)

    X, Y, Z, levels = get_kde_histogram(x_values, y_values)

    if "Z_ratio" in prop_y:
        y_edges = np.linspace(0, 4, bins_y)
    if "SFR_" in prop_y:
        y_edges = np.linspace(-2.5, 2.9, bins_y)
    if "sfr_" in prop_y:
        y_edges = np.linspace(-3, 3, bins_y)
    if "v_" in prop_y:
        y_edges = np.linspace(0, 1050, bins_y)
    if "eta" in prop_y:
        y_edges = np.linspace(0, 0.25, bins_y)
    if "M_out" in prop_y:
        y_edges = np.linspace(5.5, 10.5, bins_y)

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

    x_grid, y_grid = np.meshgrid(x_edges, y_edges)
    if statistic == "count":
        col_norm = colors.TwoSlopeNorm(vmin=0, vcenter=1.5, vmax=3)
        cmap = plt.get_cmap("plasma")
    else:
        col_norm = colors.TwoSlopeNorm(vmin=6, vcenter=8, vmax=10)
        cmap = plt.get_cmap("inferno")

    f, axs = plt.subplots(
        ncols=2,
        nrows=1,
        gridspec_kw={
            "hspace": 0.1,
            "wspace": 0.1,
            "width_ratios": [24, 1],
            "height_ratios": [24],
        },
        figsize=[10, 8],
    )
    ax = axs[0]
    cax = axs[1]
    subfig = ax.pcolormesh(x_grid, y_grid, hist.T, norm=col_norm, cmap=cmap)
    jades = get_jades_data()
    contour_color = "white" if for_slides else "black"
    ax.contour(
        X,
        Y,
        Z,
        levels=levels,
        linestyles=["solid", "dashed", "dotted"],
        colors=contour_color,
        linewidths=3,
    )
    if "v_" in prop_y:
        ax.set_ylim(0, 1050)
    color_ha = "white" if for_slides else "black"
    color_oiii = "white" if for_slides else "black"
    if prop_x == "M_star_log":
        if prop_y in {
            "M_out_log",
            "M_out_aperture_log",
            "M_out_aperture_log_03",
            "M_out_and_wind_log",
        }:
            ax.scatter(
                jades["M_star_log_Oiii"],
                jades["M_out_log_Oiii"],
                marker="*",
                s=400,
                color=color_oiii,
                edgecolors="white",
                linewidths=0.5,
                label="Jades OIII",
            )
            ax.scatter(
                jades["M_star_log_Ha"],
                jades["M_out_log_Ha"],
                marker="*",
                s=400,
                color=color_ha,
                label=r"JADES H$\alpha$",
                edgecolors="white",
                linewidths=0.5,
            )
        elif "v_" in prop_y:
            ax.scatter(
                jades["M_star_log_Oiii"],
                jades["v_out_Oiii"],
                c=jades["M_out_log_Oiii"],
                marker="*",
                s=400,
                edgecolors="white",
                linewidths=1,
                norm=col_norm,
                cmap=plt.get_cmap("inferno"),
            )
            ax.scatter(
                jades["M_star_log_Ha"],
                jades["v_out_Ha"],
                c=jades["M_out_log_Ha"],
                marker="*",
                s=400,
                label="JADES",
                edgecolors="white",
                linewidths=1,
                norm=col_norm,
                cmap=plt.get_cmap("inferno"),
            )
        elif "SFR" in prop_y:
            ax.scatter(
                jades["M_star_log_Oiii"],
                jades["SFR_log_Oiii"],
                c=jades["M_out_log_Oiii"],
                marker="*",
                s=400,
                norm=col_norm,
                cmap=plt.get_cmap("inferno"),
                edgecolors="white",
                linewidths=1,
            )
            ax.scatter(
                jades["M_star_log_Ha"],
                jades["SFR_log_Ha"],
                c=jades["M_out_log_Ha"],
                marker="*",
                s=400,
                norm=col_norm,
                cmap=plt.get_cmap("inferno"),
                label="JADES",
                edgecolors="white",
                linewidths=1.0,
            )
        if prop_y in {
            "M_out_log",
            "v_out",
            "SFR_log",
            "M_out_aperture_log",
            "M_out_aperture_log_03",
        }:
            ax.legend(fontsize=15, loc="lower right")
    if with_fits:
        z_mean = df["z"].mean()
        x = np.linspace(x_values.min(), x_values.max(), 100)
        y_speagle = (0.84 - 0.026 * get_universe_age(z_mean)) * x + (
            0.11 * get_universe_age(z_mean) - 6.51
        )
        ax.legend(fontsize=15, loc="lower right")

    if statistic == "count":
        color_label = r"$\log(n_\mathrm{galaxies})$"
    else:
        if statistic == "quantile":
            color_label = rf"$M_{{\mathrm{{out,}} {quantile}}}[\log(M_\odot)]$"
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
    if title is not None:
        ax.set_title(title, size=25)
    return


def prop_prop_histogram_overlayed(
    df,
    df_2,
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
    with_histogram=True,
    both_contours=False,
    with_labels=True,
):
    df = df.copy(deep=True)
    df_2 = df_2.copy(deep=True)
    label_colors(for_slides)
    if color_log:
        df["color_prop"] = np.log10(df[color_prop])
        color_prop = "color_prop"

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=[prop_x, prop_y], inplace=True)
    df_2.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_2.dropna(subset=[prop_x, prop_y], inplace=True)

    x_values = df[prop_x]
    y_values = df[prop_y]
    x_values_2 = df_2[prop_x]
    y_values_2 = df_2[prop_y]

    X, Y, Z, levels = get_kde_histogram(x_values_2, y_values_2, serra=True)
    if both_contours:
        X_tng, Y_tng, Z_tng, levels_tng = get_kde_histogram(x_values, y_values)

    if log_x:
        x_values = np.ma.masked_invalid(np.log10(x_values))
    if log_y:
        y_values = np.ma.masked_invalid(np.log10(y_values))

    x_edges = np.linspace(x_values.min(), x_values.max(), bins_x)
    y_edges = np.linspace(y_values.min(), y_values.max(), bins_y)
    x_edges_cont = x_edges
    y_edges_cont = y_edges

    if "Z_ratio" in prop_y:
        y_edges = np.linspace(0, 4, bins_y)
        y_edges_cont = np.linspace(0, 4, bins_y)
    if "SFR_" in prop_y:
        y_edges = np.linspace(-2, 1, bins_y)
        y_edges_cont = np.linspace(0, 4, bins_y)
    if "sfr_" in prop_y:
        y_edges = np.linspace(-3, 3, bins_y)
        y_edges_cont = np.linspace(0, 4, bins_y)
    if "v_" in prop_y:
        y_edges = np.linspace(0, 950, bins_y)
        y_edges_cont = np.linspace(0, 4, bins_y)
    if "eta" in prop_y:
        y_edges = np.linspace(0, 0.25, bins_y)
        y_edges_cont = np.linspace(0, 4, bins_y)

    hist, hist_cont, xedges_cont, yedges_cont = get_histogram_2d(
        df,
        x_values,
        y_values,
        bins=[x_edges, y_edges],
        bins_cont=[x_edges_cont, y_edges_cont],
        color_prop=color_prop,
        statistic=statistic,
        quant=quantile,
    )
    old_hist = np.copy(hist)
    if statistic == "count":
        hist = np.log10(hist)

    x_grid, y_grid = np.meshgrid(x_edges, y_edges)

    if statistic == "count":
        col_norm = colors.TwoSlopeNorm(vmin=0, vcenter=1.5, vmax=3)
        cmap = plt.get_cmap("plasma")
    else:
        col_norm = colors.TwoSlopeNorm(vmin=6, vcenter=8, vmax=10)
        cmap = plt.get_cmap("inferno")

    if with_histogram:
        f, axs = plt.subplots(
            ncols=2,
            nrows=1,
            gridspec_kw={
                "hspace": 0.1,
                "wspace": 0.1,
                "width_ratios": [24, 1],
                "height_ratios": [24],
            },
            figsize=[10, 8],
        )
        ax = axs[0]
        cax = axs[1]
        subfig = ax.pcolormesh(
            x_grid, y_grid, hist.T, norm=col_norm, cmap=cmap
        )
    else:
        f, ax = plt.subplots(
            ncols=1,
            nrows=1,
            gridspec_kw={
                "hspace": 0.1,
                "wspace": 0.1,
                "width_ratios": [24],
                "height_ratios": [24],
            },
            figsize=[9, 8],
        )

    base_cmap = plt.cm.viridis
    alphas = np.linspace(0.2, 0.8, len(levels) - 1)
    for i in range(len(levels) - 1):
        ax.contourf(
            X,
            Y,
            Z,
            levels=[levels[i], levels[i + 1]],
            colors="green",
            alpha=alphas[i],
        )
    green_block = Patch(facecolor="green", label="SERRA")
    ax.set_xlim(x_edges.min(), x_edges.max())
    ax.set_ylim(0, 2.9)

    contour_color = "white" if for_slides else "black"
    if both_contours:
        ax.contour(
            X_tng,
            Y_tng,
            Z_tng,
            levels=levels_tng,
            linestyles=["solid", "dashed", "dotted"],
            colors=contour_color,
            linewidths=3,
        )
    black_line = mlines.Line2D([], [], color=contour_color, label="TNG50")

    jades = get_jades_data()
    color_ha = "white" if for_slides else "black"
    color_oiii = "white" if for_slides else "black"

    if prop_x == "M_star_log":
        if prop_y in {
            "M_out_log",
            "M_out_aperture_log",
            "M_out_aperture_log_03",
        }:
            jades_data = ax.scatter(
                jades["M_star_log_Oiii"],
                jades["M_out_log_Oiii"],
                marker="*",
                s=400,
                color=color_oiii,
                edgecolors="white",
                linewidths=0.5,
                label="Jades",
            )
            ax.scatter(
                jades["M_star_log_Ha"],
                jades["M_out_log_Ha"],
                marker="*",
                s=400,
                color=color_ha,
                edgecolors="white",
                linewidths=0.5,
            )
        elif "v_" in prop_y:
            ax.scatter(
                jades["M_star_log_Oiii"],
                jades["v_out_Oiii"],
                c=jades["M_out_log_Oiii"],
                marker="*",
                s=400,
                edgecolors="white",
                linewidths=1,
                norm=col_norm,
                cmap=plt.get_cmap("inferno"),
            )
            jades_data = ax.scatter(
                jades["M_star_log_Ha"],
                jades["v_out_Ha"],
                c=jades["M_out_log_Ha"],
                marker="*",
                s=400,
                label="JADES",
                edgecolors="white",
                linewidths=1,
                norm=col_norm,
                cmap=plt.get_cmap("inferno"),
            )
        elif "SFR" in prop_y:
            ax.scatter(
                jades["M_star_log_Oiii"],
                jades["SFR_log_Oiii"],
                c=jades["M_out_log_Oiii"],
                marker="*",
                s=400,
                norm=col_norm,
                cmap=plt.get_cmap("inferno"),
                edgecolors="white",
                linewidths=1,
            )
            jades_data = ax.scatter(
                jades["M_star_log_Ha"],
                jades["SFR_log_Ha"],
                c=jades["M_out_log_Ha"],
                marker="*",
                s=400,
                norm=col_norm,
                cmap=plt.get_cmap("inferno"),
                label="JADES",
                edgecolors="white",
                linewidths=1.0,
            )

    if prop_y in {
        "M_out_log",
        "v_out",
        "SFR_log",
        "M_out_aperture_log",
        "M_out_aperture_log_03",
    }:
        if both_contours:
            ax.legend(
                handles=[black_line, green_block, jades_data],
                fontsize=15,
                loc="lower right",
            )
        else:
            ax.legend(
                handles=[black_line, jades_data],
                fontsize=15,
                loc="lower right",
            )

    if prop_y in "Z_ratio":
        if both_contours:
            ax.legend(
                handles=[black_line, green_block],
                fontsize=15,
                loc="upper right",
            )
        else:
            ax.legend(handles=[black_line], fontsize=15, loc="upper right")

    if with_histogram:
        if with_labels:
            for i in range(len(xedges_cont) - 1):
                for j in range(len(yedges_cont) - 1):
                    label_base = old_hist[i, j]
                    if statistic == "count":
                        count = int(label_base)
                    else:
                        count = f"{label_base:.2f}"
                    if label_base > 0:
                        ax.text(
                            xedges_cont[i]
                            + (xedges_cont[i + 1] - xedges_cont[i]) / 2,
                            yedges_cont[j]
                            + (yedges_cont[j + 1] - yedges_cont[j]) / 2,
                            count,
                            color="blue",
                            ha="center",
                            va="center",
                        )
        if statistic == "count":
            color_label = r"$\log(n_\mathrm{galaxies})$"
        else:
            if statistic == "quantile":
                color_label = (
                    rf"$M_{{\mathrm{{out,}} {quantile}}}[\log(M_\odot)]$"
                )
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
        ax.tick_params(labelsize=25)

    ax.set_xlabel(prop_labels(prop_x), size=25)
    ax.set_ylabel(prop_labels(prop_y), size=25)
    ax.tick_params(labelsize=20)
    if title is not None:
        ax.set_title(title, size=25)
    return


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

    df.dropna(subset=prop_x, inplace=True)
    df.dropna(subset=prop_y, inplace=True)

    x_values = df[prop_x]
    y_values = df[prop_y]
    if log_x:
        x_values = np.ma.masked_invalid(np.log10(x_values))
    if log_y:
        y_values = np.ma.masked_invalid(np.log10(y_values))

    f, ax = plt.subplots(
        ncols=1,
        nrows=1,
        gridspec_kw={
            "hspace": 0.1,
            "wspace": 0.1,
            "width_ratios": [24],
            "height_ratios": [24],
        },
        figsize=[10, 8],
    )
    color = "r" if for_slides else "b"
    ax.scatter(
        x_values, y_values, s=1, color=color, alpha=0.3, label="TNG data"
    )
    ax.tick_params(labelsize=15)
    ax.legend(fontsize=15)
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
            "wspace": 0.1,
            "width_ratios": [24],
            "height_ratios": [24],
        },
        figsize=[10, 8],
    )
    for i, element in enumerate(fractions):
        ax.plot(
            centers,
            element,
            label=rf"$10^{{{thresholds[i]}}} M_\odot$ threshold",
        )
    ax.tick_params(labelsize=15)
    ax.legend(fontsize=15)
    ax.set_xlabel(prop_labels("M_out_log"), size=25)
    ax.set_ylabel("Fraction of detectable outflows", size=25)
    return


def plot_galaxy_evolution(
    galaxies,
    prop_x,
    prop_y,
    color_prop,
    prop_y2=None,
    for_slides=False,
    title=None,
    sample=None,
):
    if sample == "small":
        path = "/ptmp/mpa/ivkos/outflows/history_2.pickle"
        with open(path, "rb") as f:
            galaxy = pickle.load(f)
        rel_galaxies = {}
        for key in galaxy.keys():
            if np.sum(~np.isnan(galaxy[key]["M_out/M_star"])) > 12:
                if galaxy[key]["M_out/M_star"][11] > 5.2:
                    rel_galaxies[key] = galaxy[key]
        del rel_galaxies[153887]
        galaxies = rel_galaxies
    elif sample == "large":
        path = "/ptmp/mpa/ivkos/outflows/massive_galaxy.pickle"
        with open(path, "rb") as f:
            galaxy = pickle.load(f)

    label_colors(for_slides)
    if color_prop is not None:
        f, axs = plt.subplots(
            ncols=1,
            nrows=2,
            gridspec_kw={
                "hspace": 0.4,
                "wspace": 0.1,
                "width_ratios": [24],
                "height_ratios": [1, 24],
            },
            figsize=[10, 10],
        )
        col_norm = colors.TwoSlopeNorm(vmin=-9, vcenter=-8.25, vmax=-7.5)
        cax = axs[0]
        ax = axs[1]
    else:
        f, ax = plt.subplots(figsize=[10, 8])
        cax = None
    if color_prop == "sSFR_log":
        col_norm = colors.TwoSlopeNorm(vmin=-9, vcenter=-8.5, vmax=-8)
    elif color_prop == "M_star_log":
        col_norm = colors.TwoSlopeNorm(vmin=9, vcenter=10, vmax=11)

    ax_right = ax.twinx() if prop_y2 else None
    linestyle = ["solid", "dashed", "dotted", "dashdot"]
    for i, (id, galaxy) in enumerate(galaxies.items()):
        if len(galaxies.items()) > 1:
            ax.plot(
                galaxy[prop_x],
                galaxy[prop_y],
                linewidth=2,
                linestyle=linestyle[i],
                label=f"$M_{{\\star, \\mathrm{{fin}}}}=10^{{{galaxy['M_star_log'][0]:.1f}}}M_\\odot$",
            )
        else:
            ax.plot(
                galaxy[prop_x],
                galaxy[prop_y],
                linewidth=2,
                color="red",
                label=f"$M_{{\\star, \\mathrm{{fin}}}}=10^{{{galaxy['M_star_log'][0]:.1f}}}M_\\odot$",
            )
        if color_prop is not None:
            subfig = ax.scatter(
                galaxy[prop_x],
                galaxy[prop_y],
                marker="*",
                c=galaxy[color_prop],
                cmap=plt.get_cmap("plasma"),
                s=500,
                norm=col_norm,
            )
        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        ax_top.set_xticks(galaxy[prop_x][::2])
        ax_top.set_xticklabels([f"{zi:.1f}" for zi in galaxy["z"][::2]])
        ax_top.tick_params(axis="x", labelsize=15)
        ax_top.set_xlabel("redshift $z$", fontsize=20)

        if prop_y2 and ax_right:
            if prop_y2 == "BH_mdot_log":
                y2 = galaxy[prop_y2] - 9
            else:
                y2 = galaxy[prop_y2]
            ax_right.plot(
                galaxy[prop_x],
                y2,
                linewidth=2,
                linestyle="dashed",
                color="blue",
            )

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
        ax_right.tick_params(axis="y", labelcolor="blue", labelsize=15)

    ax.tick_params(labelsize=15)
    ax.set_xlabel(prop_labels(prop_x), size=25)
    if len(galaxies.items()) > 1:
        ax.set_ylabel(prop_labels(prop_y), size=25)
        ax.tick_params(axis="y", labelsize=15)
    else:
        ax.set_ylabel(prop_labels(prop_y), size=25, color="red")
        ax.tick_params(axis="y", labelsize=15, labelcolor="red")

    ax.legend(fontsize=15)
    return
