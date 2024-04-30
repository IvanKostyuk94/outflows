import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import norm
from sklearn.mixture import GaussianMixture as GMM
from Grid_halo import map_to_new_dict


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
    lin_data = ["Flow_Velocities", "Rot_Velocities"]
    log_data = ["Coordinates", "Temperature", "StarFormationRate"]
    if key in log_data:
        # Add regulator to data to avoid values of -inf
        data = np.log(data + np.min(np.abs(data)) / 1e5)
    elif key in lin_data:
        pass
    else:
        raise NotImplementedError(
            f"Normalization for {key} is not implemented yet."
        )

    normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
    return normalized


def get_data(gas, keys):
    if len(keys) == 1:
        data = gas[keys[0]]
        data = data.reshape(-1, 1)
    else:
        data = []
        for key in keys:
            if gas[key].ndim == 1:
                data.append(normalize(gas[key], key))
            elif gas[key].ndim == 2:
                for i in range(gas[key].shape[1]):
                    data.append(normalize(gas[key][:, i], key))
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


def associate_gas_to_peaks(gas, n_peaks, props):
    gmm = GMM(
        n_components=n_peaks,
        max_iter=5000,
        covariance_type="full",
        random_state=42,
    )
    data = get_data(gas, keys=props)
    gmm.fit(data)
    # means = gmm.fit(x).means_
    # stds = np.sqrt(gmm.fit(x).covariances_)
    # weights = gmm.fit(x).weights_
    # pdfs = []
    # for i in range(n_peaks):
    #     pdfs.append(stats.norm(loc=means[i][0], scale=stds[i][0][0]))
    # probs = get_probs(gas["Flow_Velocities"], pdfs, weights)
    # gas["group"] = np.argmax(probs, axis=0) + 1
    gas["group"] = gmm.predict(data)
    return


def group_gas(
    gas,
    props=["Flow_Velocities"],
    min_number=1,
    max_number=10,
    peak_number=None,
):
    if peak_number is None:
        n_peak, bics, counters = select_number_of_peaks(
            gas, keys=props, min_number=min_number, max_number=max_number
        )
        print(f"Selecting {n_peak} peaks. The bic values obtained were {bics}")
    else:
        n_peak = peak_number
    associate_gas_to_peaks(gas, n_peaks=n_peak, props=props)
    return n_peak


def select_galaxy_group(group_array):
    count = 0
    galaxy_group = 0
    for i, group in enumerate(group_array):
        if group["count"] > count:
            galaxy_group = i
    return galaxy_group


def get_only_outflowing_gas(out_gas, galaxy_group):
    idces_rel_gas = out_gas["group"] != galaxy_group
    rel_gas = map_to_new_dict(out_gas, idces_rel_gas)
    return rel_gas
