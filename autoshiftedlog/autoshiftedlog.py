import scipy.stats
import numpy as np
from numpy import sign, abs, exp, log, pi, sqrt
from numpy import nanmean as mean, nanstd as std, nanmedian as median, nanmin as min, nanmax as max

# .95 quantile of Extreme Value Distribution
_gumble_p95 = scipy.stats.gumbel_l.ppf(.95)

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
    gumble_p95 = _gumble_p95  # cache this call for speed
    a_n = (2*log(n))**(-0.5)
    b_n = (2*log(n) - log(log(n)) - log(4*pi))**0.5
    threshold = gumble_p95 * a_n + b_n

    # TODO: Warnings are issued here when vector contains nans.
    vector[vector > threshold] = threshold
    vector[vector < -threshold] = -threshold
    return vector

def _get_data_range(vector):
    IQR = scipy.stats.iqr(vector, nan_policy='omit')
    # TODO: Scipy computes IQR slightly differently
    # from MATLAB. This leads to slightly different
    # results from Marron's version

    if IQR == 0:
        data_range = max(vector) - min(vector)
    else:
        data_range = IQR

    return data_range

def _shiftedlog(vector, shift, _data_range=None):
    "Apply shifted log transformation with the given shift"
    beta = sign(shift) * (exp(abs(shift))-1)

    if _data_range is not None:
        data_range = _data_range
    else:
        data_range = _get_data_range(vector)

    # Transform data based on sign of beta
    if beta == 0:
        vector = vector
    elif beta > 0:
        alpha = abs(1.0/beta)
        vector = log(vector - min(vector) + alpha*data_range)
    else:
        alpha = abs(1.0/beta)
        vector = -log(max(vector) - vector + alpha*data_range)

    vector_median = median(vector)

    MAD = mean(abs(vector - vector_median)) * sqrt(pi / 2)

    if MAD == 0:
        # if the MAD is 0, just return zeroes but retain nans.
        vector[~np.isnan(vector)] = 0
        return vector

    vector = (vector - vector_median) / MAD
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

    data_range = _get_data_range(vector)  # computing this in advance speeds up the search

    # Set up an array of possible shift values to try
    if _skew(vector) > 0:
        shifts = np.arange(0.0, 9.0, step=0.01)
    else:
        shifts = -np.arange(0.0, 9.0, step=0.01)

    # Find the shift that minimizes the desired score function
    scores = [score(_shiftedlog(vector, s, data_range)) for s in shifts]

    minimizing_index = np.argmin(scores)

    best_shift = shifts[minimizing_index]
    best_transformation = _shiftedlog(vector, best_shift, data_range)

    if verbose:
        best_beta = sign(best_shift) * (exp(abs(best_shift))-1)
        print("Transformation parameter beta: {}".format(best_beta))

    return best_transformation
