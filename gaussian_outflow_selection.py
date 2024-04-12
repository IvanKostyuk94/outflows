import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import norm
from sklearn.mixture import GaussianMixture as GMM


def select_number_of_peaks(gas):
    x = gas["Flow_Velocities"]
    x = x.reshape(-1, 1)

    bics = []
    min_bic = 0
    counter = 1
    num_bics = 5
    for _ in range(num_bics):
        gmm = GMM(
            n_components=counter,
            max_iter=5000,
            random_state=0,
            covariance_type="full",
        )
        _ = gmm.fit(x).predict(x)
        bic = gmm.bic(x)
        bics.append(bic)
        if bic < min_bic or min_bic == 0:
            min_bic = bic
            opt_bic = counter
        counter = counter + 1
    return opt_bic, bics


def get_probs(v_out, pdfs, weights):
    return np.array(
        [pdf.pdf(v_out) * weight for pdf, weight in zip(pdfs, weights)]
    )


def associate_gas_to_peaks(gas, n_peaks):
    gmm = GMM(
        n_components=n_peaks,
        max_iter=5000,
        random_state=10,
        covariance_type="full",
    )
    x = gas["Flow_Velocities"]
    x = x.reshape(-1, 1)
    means = gmm.fit(x).means_
    stds = np.sqrt(gmm.fit(x).covariances_)
    weights = gmm.fit(x).weights_
    pdfs = []
    for i in range(n_peaks):
        pdfs.append(stats.norm(loc=means[i][0], scale=stds[i][0][0]))
    probs = get_probs(gas["Flow_Velocities"], pdfs, weights)
    gas["group"] = np.argmax(probs, axis=0) + 1
    return


def group_gas(gas):
    opt_bic, bics = select_number_of_peaks(gas)
    print(f"Selecting {opt_bic} peaks. The bic values obtained were {bics}")
    associate_gas_to_peaks(gas, n_peaks=opt_bic)
    return opt_bic
