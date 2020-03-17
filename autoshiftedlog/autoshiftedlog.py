import scipy.stats
import numpy as np
from numpy import sign, abs, exp, log, pi, sqrt
from numpy import nanmean as mean, nanstd as std, nanmedian as median

def _skew(vector):
    return scipy.stats.skew(vector, nan_policy='omit')

def _andersondarling(vector):
    "Compute Anderson Darling statistic of given vector"
    vector = vector[~np.isnan(vector)]  # remove nans
    n = len(vector)

    if n < 7:
        raise ValueError('Anderson Darling statistic requires at least 7 non-NAs')

    vector = np.sort(vector)

    f = scipy.stats.norm.cdf(vector, mean(vector), std(vector, ddof=1))
    i = np.arange(1, n+1)
    S = sum((2*i - 1)/n * (log(f) + log(1-np.flip(f))))

    return -n-S

def _winsorize(vector):
    "Winsorize based on 95th percentile of extreme value distribution"
    n = np.count_nonzero(~np.isnan(vector))
    gumble_p95 = scipy.stats.gumbel_l.ppf(.95)
    a_n = (2*log(n))**(-0.5)
    b_n = (2*log(n) - log(log(n)) - log(4*pi))**0.5
    threshold = gumble_p95 * a_n + b_n

    # TODO: Warnings are issued here when vector contains nans.
    vector[vector > threshold] = threshold
    vector[vector < -threshold] = -threshold
    return vector

def _shiftedlog(vector, shift):
    "Apply shifted log transformation with the given shift"
    n = np.count_nonzero(~np.isnan(vector))
    beta = sign(shift) * (exp(abs(shift))-1)

    IQR = scipy.stats.iqr(vector, nan_policy='omit')
    # TODO: Scipy computes IQR slightly differently
    # from MATLAB. This leads to slightly different
    # results from Marron's version

    if IQR == 0:
        data_range = max(vector) - min(vector)  # NOTE: this is nan-safe
    else:
        data_range = IQR

    # Transform data based on sign of beta
    if beta == 0:
        vector = vector
    elif beta > 0:
        alpha = abs(1.0/beta)
        vector = log(vector - min(vector) + alpha*data_range)
    else:
        alpha = abs(1.0/beta)
        vector = -log(max(vector) - vector + alpha*data_range)

    MAD = mean(abs(vector - median(vector))) * sqrt(pi / 2)

    if MAD == 0:
        # if the MAD is 0, just return zeroes but retain nans.
        vector[~np.isnan(vector)] = 0
        return vector

    vector = (vector - median(vector)) / MAD
    vector = _winsorize(vector)
    vector = (vector - mean(vector)) / std(vector, ddof=1)

    return vector

def autoshiftedlog(vector, score_function='Anderson Darling', verbose=False):
    """Apply shifted log transformation, automatically selecting the best shift
       based on desired score function.

       vector: a numpy array or pandas Series
       score_function: 'Anderson Darling' or 'skewness'
       verbose: if True, prints the optimal value of beta
    """

    if score_function == 'Anderson Darling':
        score = _andersondarling

    elif score_function == 'skewness':
        score = _skew

    else:
        raise ValueError("metric must be 'Anderson Darling' or 'skewness'")

    if std(vector) == 0:
        # if the SD is 0, just return zeroes but retain nans.
        vector[~np.isnan(vector)] = 0
        return vector

    # Set up an array of possible shift values to try
    if _skew(vector) > 0:
        shifts = np.arange(0.0, 9.0, step=0.01)
    else:
        shifts = -np.arange(0.0, 9.0, step=0.01)

    # Find the shift that minimizes the desired score function
    scores = [score(_shiftedlog(vector, s)) for s in shifts]

    minimizing_index = np.argmin(scores)

    best_shift = shifts[minimizing_index]
    best_transformation = _shiftedlog(vector, best_shift)

    if verbose:
        best_beta = sign(best_shift) * (exp(abs(best_shift))-1)
        print("Transformation parameter beta: {}".format(best_beta))

    return best_transformation
