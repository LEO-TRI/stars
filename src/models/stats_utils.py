import numpy as np
import pandas as pd
from scipy import stats

def weighted_avg_array(data : np.ndarray, weights : np.ndarray) -> np.ndarray:
    """
    Calculates a weighted average on a np.array
    
    Works over 2D array. In this case, each column in data will be combined with 
    the corresponding column in weights and average will be produced over axis 0

    Parameters
    ----------
    data : np.ndarray
        The data to average
    weights : np.ndarray
        The weights used, must be the same size as data

    Returns
    -------
    np.ndarray
        The weighted average. Can be a float if arguments are 1D and a 1D array if
        arguments are 2D

    Raises
    ------
    ValueError
        If data and weights dont have the same dim        
    """

    if (len(data) == len(weights)) is not True:
        raise ValueError("data and weights must be the same length on axis 0")

    data = np.asarray(data)
    weights = np.asarray(weights)

    #Broadcast the weight array to the same dims as data, and then remove for each column 
    #the corresponding missing vals in data in weights and replace with 0. If data is 1D
    #then weights doesn't need to be broadcasted since it is already 1D. 

    nan_indices = np.isnan(data)


    #(data.shape[1], 1) broadcast the 1D array data.shape[1] times along the rows. Taking
    #the transpose creates a weights array where all columns are identical
    if (data.ndim > 1) & (weights.ndim == 1):
        weights = np.tile(weights, (data.shape[1], 1)).T

    data[nan_indices] = 0
    weights[nan_indices] = 0 

    #Manual dot product so as to generate the diagonal results of the output matrix / sum of weights

    return np.sum(data * weights, axis = 0) / np.sum(weights, axis = 0)

def weighted_avg_pandas(df : pd.DataFrame, data_col : str = None, weigth_col: str = "weights") -> pd.DataFrame:
    """
    Computes a weighted average using pandas

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe
    data_col : str, optional
        The column(s) on which to compute the avg, by default None
        Can be several columns
    weigth_col : str, optional
        the column of weights, by default "weights"

    Returns
    -------
    pd.Series
        A Series with for each column in data_col the corresponding avg
    """

    weights = df[weigth_col]
    
    if data_col is None: 
        df = df.select_dtypes(np.number).drop(columns = weigth_col)
        return df
    else:
        df = df[data_col]

    return (df * weights).sum() / weights.sum()

def calculate_weighted_average(group, return_col='sectorReturns', weight_col='fmv_prop'):
    
    return (group[return_col] * group[weight_col]).sum() / group[weight_col].sum()

def annual_returns(data : pd.DataFrame) -> pd.DataFrame:
    """
    Function to calculate annualized returns
    
    Can be used with an apply in a groupby
    e.g. df.groupby("year).apply(annual_returns)

    Parameters
    ----------
    data : pd.DataFrame
        The data to transform, each column will be annualized
        All columns must be numerical dtype

    Returns
    -------
    pd.DataFrame
        The annualized df
    """

    return (1 + data).prod() - 1

def real_returns(nominal_returns : np.ndarray, inflation : np.ndarray) -> np.ndarray:
    """
    Calculate the returns based on an array of nominal returns and an array of inflation
    
    nominal_returns and inflation must have the same length

    Parameters
    ----------
    nominal_returns : np.ndarray
    inflation : np.ndarray

    Returns
    -------
    np.ndarray
        The real returns
    """

    return ((1 + nominal_returns) / (1 + inflation)) - 1 

def cum_returns(series: np.ndarray, name: str = "SafeReturns") -> np.ndarray:
    """
    Calculate safe returns over time based on a given series of percentage changes.

    Parameters
    ----------
    series : np.ndarray
        An array containing percentage changes over time.
    name : str, optional, default: "SafeReturns"
        The name to be assigned to the resulting Series.

    Returns
    -------
    pd.Series
        A 1-dimensional pd.Series representing safe returns over time.

    Notes
    -----
    - The input series should contain percentage changes.
    - The function calculates safe returns based on a cumulative product of (1 + percentage_change/100).
    - The resulting array represents the cumulative safe returns over time.
    """
    series[0] = 0  # Replacing the first value since at t=0 there is no interest yet
    val = 1 + series/100
    val_cumprod = np.cumprod(val)
    result_returns = val_cumprod * 100

    return pd.Series(result_returns, name=name)

def kde_estimator(data: np.ndarray, lower_bound: float, upper_bound: float) -> np.ndarray:
    """
    TODO : Move to a stats & ML file

    Perform Kernel Density Estimation (KDE) on a given dataset within a specified range.

    Parameters
    ----------
    data : numpy.ndarray
        The input dataset for which KDE will be estimated.
    lower_bound : float
        The lower bound of the range for KDE estimation.
    upper_bound : float
        The upper bound of the range for KDE estimation.

    Returns
    -------
    np.ndarray
      An array of data produced by a KDE estimator object based on the input data within the specified range.
    """

    kde = stats.gaussian_kde(data)
    return kde(np.linspace(lower_bound, upper_bound, len(data)))


def cumulative_mean(series):
    return np.cumsum(series) / np.arange(1, len(series)+1, 1)


def cumulative_quantile(series, q=.1, start=10):
    """
    :return: tuple, x the number of simulations, and quantiles list
    """
    x_axis = np.arange(start, len(series)+1, 1)
    quantiles = []
    assert start < len(
        series), 'start is less than series, provide a longer series'
    for i in x_axis:
        sub_series = series[:i]
        quantile = np.quantile(sub_series, q)
        quantiles.append(quantile)
    return x_axis, quantiles


def percentile05(x): 
    return np.percentile(x, q=5)


def bootstrap_estimation(series, estimator, n_samples=100, n_repeat=500):
    """
    Samples n_samples samples from the series, n_repeat times, to estimate the estimator on the sample
    """
    results = []
    for iteration in range(n_repeat):
        sample = series.sample(n_samples, replace=True)
        results.append(estimator(sample))
    return np.array(results)


def get_timeseries_bootstrap_estimation(series, estimator, n_samples_list, n_repeat=500, ci_low=5, ci_high=95):
    """
    for different sample size, runs the bootstrap_estimation function
    """
    percentile_estimates = []
    confidence_interval_low = []
    confidence_interval_high = []
    for n_sample in n_samples_list:
        results = bootstrap_estimation(
            series, estimator, n_samples=n_sample, n_repeat=n_repeat)
        percentile_estimates.append(np.median(results))
        confidence_interval_low.append(np.percentile(results, q=ci_low))
        confidence_interval_high.append(np.percentile(results, q=ci_high))

    return pd.DataFrame({
        f"ci {ci_low}": confidence_interval_low,
        f"ci {ci_high}": confidence_interval_high,
        "estimator": percentile_estimates,
        "n_samples": list(n_samples_list)
    })

def get_summary_statistics(series) -> pd.DataFrame:
    """
    Function to generate a summary statistics reports for a series

    Parameters
    ----------
    series : array_like
        An array_like object of numbers 

    Returns
    -------
    pd.DataFrame
        A dataframe with 1 row and 10 columns containing 10 summary statistics

    Raises
    ------
    TypeError
        Raises an error if the input is not array_like
    """
    #Check that the argumment is array_like and convert to a pd.Series

    if isinstance(series, (np.ndarray, list, tuple)):
        series = pd.Series(series)
    elif isinstance(series, pd.DataFrame):
        if (series.shape[1] == 1):
            series = pd.Series(series.values.ravel())
    else: 
        raise TypeError("Input needs to be a 1D array_like object")

    summary_statistics = pd.DataFrame({
        "min": [series.min()],
        "max": [series.max()],
        "VaR 99%": [series.quantile(.01, interpolation='lower')],
        "VaR 95%": [series.quantile(.05, interpolation='lower')],
        "VaR 90%": [series.quantile(.1, interpolation='lower')],
        "mean": [series.mean()],
        "std": [series.std()],
        "median": [series.median()],
        "skew": [series.skew()],
        "kurt": [series.kurt()],
    })
    
    return summary_statistics
