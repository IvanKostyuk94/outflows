import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from .config import prop_labels


def get_jades_data():
    data = {}
    data["M_star_log_Oiii"] = np.array([7.69, 7.60, 7.85, 8.09, 7.78, 8.63])
    data["M_star_log_Ha"] = np.array(
        [8.54, 8.11, 7.73, 8.28, 7.81, 7.93, 7.85, 8.24]
    )
    data["SFR_log_Oiii"] = np.array([0.09, 0.53, 0.61, 0.39, 0.41, 1.14])
    data["SFR_log_Ha"] = np.array(
        [0.65, 0.74, 0.14, 0.34, 0.09, -0.67, 0.61, 0.01]
    )
    data["M_out_log_Oiii"] = np.array([6.46, 7.07, 6.84, 6.56, 7.12, 8.26])
    data["M_out_log_Ha"] = np.array(
        [6.74, 7.17, 6.00, 6.54, 6.03, 5.85, 6.67, 6.51]
    )
    data["v_out_Oiii"] = np.array([500, 234, 701, 401, 259, 289])
    data["v_out_Ha"] = np.array([267, 444, 497, 229, 275, 648, 261, 911])
    return data


def plot_jades_hist():
    jades = get_jades_data()
    m_star = np.concatenate((jades["M_star_log_Oiii"], jades["M_star_log_Ha"]))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(
        m_star,
        density=False,
        alpha=0.8,
    )
    ax.set_xlabel(r"$\log(M_\star/M_\odot)$", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis="both", which="major", labelsize=12)
    plt.tight_layout()
    plt.show()
    return
