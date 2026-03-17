import logging
import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from utils import map_to_new_dict

logger = logging.getLogger(__name__)


def get_opt_bic(data, min_number, max_number):
    bics = []
    counters = []
    min_bic = 0
    counter = 1
    for _ in range(min_number, max_number + 1):
        gmm = GMM(
            n_components=counter,
            max_iter=5000,
            random_state=42,
            covariance_type="full",
        )
        _ = gmm.fit(data).predict(data)
        bic = gmm.bic(data)
        bics.append(bic)
        if bic < min_bic or min_bic == 0:
            min_bic = bic
            opt_bic = counter
        counter = counter + 1
        counters.append(counter)
    return opt_bic, bics, counters


def normalize(data, key):
    lin_data = [
        "Flow_Velocities",
        "Rot_Velocities",
        "Angular_Velocities",
        "Abs_Coordinates",
    ]
    log_data = [
        "Temperature",
        "StarFormationRate",
        "Relative_Distances",
    ]
    if key in log_data:
        data = np.log(data + np.min(data[data > 0]) / 1e5)
    elif key in lin_data:
        pass
    elif key == "Coordinates":
        data = data - np.min(data)
        data = np.log(data + np.min(data[data > 0]) / 1e5)
    else:
        raise NotImplementedError(
            f"Normalization for {key} is not implemented yet."
        )
    normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
    return normalized


def get_data(gas, keys):
    if (len(keys) == 1) and (gas[keys[0]].ndim == 1):
        data = gas[keys[0]]
        data = data.reshape(-1, 1)
    else:
        data = []
        for i, key in enumerate(keys):
            if gas[key].ndim == 1:
                data.append(normalize(gas[key], key))
            elif gas[key].ndim == 2:
                for j in range(gas[key].shape[1]):
                    data.append(normalize(gas[key][:, j], key))
            else:
                raise NotImplementedError(
                    f"Can't handle data with dim {gas[key].ndim}"
                )
        data = np.array(data).T
    return data


def select_number_of_peaks(gas, keys, min_number, max_number):
    data = get_data(gas=gas, keys=keys)
    opt_bic, bics, counter = get_opt_bic(
        data=data, min_number=min_number, max_number=max_number
    )
    return opt_bic, bics, counter


def get_probs(v_out, pdfs, weights):
    return np.array(
        [pdf.pdf(v_out) * weight for pdf, weight in zip(pdfs, weights)]
    )


def associate_gas_to_peaks(gas, n_peaks, props, mass_weighted=False):
    if props is None:
        props = [
            "Flow_Velocities",
            "StarFormationRate",
            "Coordinates",
        ]
    gmm = GMM(
        n_components=n_peaks,
        max_iter=5000,
        covariance_type="full",
        random_state=42,
    )
    data = get_data(gas, keys=props)
    if mass_weighted:
        masses = gas["Masses"]
        probabilities = masses / np.sum(masses)
        n_samples = len(masses)
        indices = np.random.choice(
            np.arange(n_samples), size=n_samples, replace=True, p=probabilities
        )
        data_resampled = data[indices]
    else:
        data_resampled = data
    gmm.fit(data_resampled)
    gas["group"] = gmm.predict(data)


def group_gas(
    gas,
    mass_weighted=False,
    props=None,
    min_number=1,
    max_number=10,
    n_peak=None,
):
    if props is None:
        props = ["Flow_Velocities"]
    if n_peak is None:
        n_peak, bics, counters = select_number_of_peaks(
            gas, keys=props, min_number=min_number, max_number=max_number
        )
        logger.info("Selected %d peaks. BIC values: %s", n_peak, bics)
    associate_gas_to_peaks(
        gas, n_peaks=n_peak, props=props, mass_weighted=mass_weighted
    )
    return n_peak


def select_galaxy_group(group_array, use_weighted_distance=False, test=False):
    median_dist_min = np.inf
    median_vel_min = np.inf
    galaxy_group = 0
    if test:
        for i, group in enumerate(group_array):
            mean_vel = np.mean(group["Flow_Velocities"])
            if mean_vel < median_vel_min:
                galaxy_group = i
                median_vel_min = mean_vel
    else:
        for i, group in enumerate(group_array):
            if use_weighted_distance:
                median_dist = np.average(
                    group["Relative_Distances"], weights=group["Masses"]
                )
            else:
                median_dist = np.median(group["Relative_Distances"])
            if median_dist < median_dist_min:
                galaxy_group = i
                median_dist_min = median_dist
    return galaxy_group


def get_only_outflowing_gas(out_gas, galaxy_group, crit_vout):
    idces_rel_gas = (out_gas["group"] != galaxy_group) | (
        out_gas["Flow_Velocities"] > crit_vout
    )
    return map_to_new_dict(out_gas, idces_rel_gas)
