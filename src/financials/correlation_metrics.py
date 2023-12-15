import pandas as pd
import numpy as np

from .returns_mixin import ShiftReturnsMixin
from .financial_graphs import _corr_plot


class RollingCorrelation(ShiftReturnsMixin):
    def __init__(self, 
                 roll_corr: np.ndarray, 
                 window_size : int=4,
                 column_names: list = None, 
                 index_names: list = None) -> "RollingCorrelation":
        """
        Initialize the RollingCorrelation object.

        Holds the rolling correlation between Ardian fund return and passed public indexes

        Should always be instantiated via class method from_data()

        Parameters:
        -------
        roll_corr: np.ndarray
            The rolling correlation. Each column is the corr between Ardian returns and 1 public index
        window_size: int
            The window size for the rolling correlation
        column_names: array_like
            Names of the columns for the correlation DataFrame, by default None        
        index_names: array_like
            Names of the indexes for the correlation DataFrame, by default None

        """
        self.data = roll_corr
        self.column_names = column_names
        self.index_names = index_names
        self.window_size = window_size
        self.fig = None

    @classmethod
    def from_data(cls,
                  target_series: np.ndarray,
                  feature_series: np.ndarray,
                  window_size: int,
                  is_return: bool = False,
                  show_plot: bool = False) -> "RollingCorrelation":
        """
        Calculates rolling correlation and stores results in the class

        Instantiates an instance of the class RollingCorrelation

        If show_plot==True, produces a plotly line plot of the correlations

        Parameters
        ----------
        target_series : np.ndarray
            The series against which all correlations will be computed
        feature_series : np.ndarray
            Correlation will be computed between target_series and feature_series
        window_size: int
            The window size for the rolling correlation
        is_return: bool
             Whether feature_series is an array of absolute values or returns. If False, compute log returns of values. By default False
        show_plot: bool
             Whether to produce a plotly plot and store it as self.fig, by default False

        Returns
        -------
        rollcorr : RollingCorrelation
            An instance of the class
        """

        feature_series, column_names, index_names = RollingCorrelation.check_values(feature_series)
        iterations = RollingCorrelation.check_iterations(feature_series)

        if not is_return:
            feature_series = RollingCorrelation.log_returns(feature_series)

        target_series = np.array(target_series)
        res = np.empty((len(target_series), iterations)) * np.nan

        for i in range(iterations):
            tmp_feature = feature_series[np.isfinite(feature_series[:, i]), i]

            index_slice = len(target_series) - len(tmp_feature)
            rolling_corr = (pd.Series(target_series[index_slice:])
                            .rolling(window=window_size)
                            .corr(pd.Series(tmp_feature))
                            )
            res[index_slice:, i] = rolling_corr

        rollcorr = cls(roll_corr=res[window_size:, :],
                       window_size=window_size,
                       column_names=column_names,
                       index_names=index_names[window_size:],
                       )

        fig = rollcorr.plot()
        if show_plot:
            fig.show()

        return rollcorr

    def plot(self):
        """
        Instance method to plot the line plots of correlation.

        Stores results as self.fig

        Returns:
        - None
        """

        res = pd.DataFrame(self.data)

        if self.column_names is not None:
            res.columns = self.column_names
        if self.index_names is not None:
            res.index = self.index_names

        x_values = res.index

        res = pd.melt(res, ignore_index=False)
        # Calls on _corr_plot from financial_graphs.py to calculate lineplots
        self.fig =  _corr_plot(res, x_values=x_values)

        return self.fig

    @staticmethod
    def check_values(returns: np.ndarray, column_names: list = None, index_names: list = None) -> tuple:
        """
        Check and format input data for consistent handling.

        Parameters
        ----------
        returns : numpy.ndarray, list, or pandas.DataFrame
            The input data to be checked and formatted.
        column_names : list, optional, default: None
            List of column names for DataFrame input.
        index_names : list, optional, default: None
            List of index names for DataFrame input.

        Returns
        -------
        tuple
            A tuple containing the checked and formatted returns, column names, and index names.

        Raises
        ------
        ValueError
            If the input data is not a numpy array, list of lists of equal size, or a pandas DataFrame.
        """

        if isinstance(returns, pd.DataFrame):
            column_names = returns.columns.tolist()
            index_names = returns.index.tolist()
            returns = returns.to_numpy()

        elif isinstance(returns, list):
            returns = np.asarray(returns)

        elif not isinstance(returns, np.ndarray):
            raise ValueError(
                "Input data must be a numpy array, list of lists of equal size or a pandas DataFrame."
            )

        return returns, column_names, index_names

    @staticmethod
    def check_iterations(returns: np.ndarray) -> int:
        """
        Check and format the number of iterations based on input data.

        Parameters
        ----------
        returns : numpy.ndarray
            The input data representing returns.

        Returns
        -------
        int
            The number of iterations.

        Raises
        ------
        ValueError
            If the input data has more than 2 dimensions.
        """

        if len(returns.shape) == 2:
            iterations = returns.shape[1]
        elif len(returns.shape) == 1:
            # To allow 1 iteration on dimension y
            returns = returns.reshape(-1, 1)
            iterations = 1
        else:
            raise ValueError("Passed data must have maximum 2 dims")

        return iterations


class SignCorrelation(RollingCorrelation):
    def __init__(self, cum_corr, column_names: list = None, index_names: list = None) -> None:
        """
        Inherits shifts_return, check_values, check_iterations and plot from parent class RollingCorrelation

        Parameters
        ----------
        cum_corr : numpy.ndarray
            The cumulative correlation matrix.
        column_names : list, optional, default: None
            List of column names for the correlation matrix.
        index_names : list, optional, default: None
            List of index names for the correlation matrix.

        Returns
        -------
        None
            The function initializes the CumulativeCorrelation instance with the provided data.

        Notes
        -----
        - The input cumulative correlation matrix should be a numpy array.
        - The column_names and index_names are optional and can be provided for better identification of matrix elements.
        """

        self.data = cum_corr
        self.column_names = column_names
        self.index_names = index_names
        self.fig = None

    @classmethod
    def from_data(cls,
                  target_series: np.ndarray,
                  feature_series: np.ndarray,
                  is_return: bool = False,
                  show_plot: bool = False) -> "SignCorrelation":
        """
        Calculate correlations by converting positive increases to 1 and decreases to -1, and then computing a cumulative sum.

        This class method instantiates an instance of the SignCorrelation class, providing a convenient way to compute correlations
        between a target series and a feature series with optional parameters for handling returns and displaying a plot.

        Parameters
        ----------
        target_series : numpy.ndarray
            The series against which all correlations will be computed.
        feature_series : numpy.ndarray
            The series for which correlations with the target series will be calculated.
        is_return : bool, optional, default: False
            If True, assumes that the input series are returns, and the cumulative sum will be computed accordingly.
        show_plot : bool, optional, default: False
            If True, displays a plot of the cumulative correlation.

        Returns
        -------
        s_corr : SignCorrelation
            An instance of the SignCorrelation class representing the calculated correlations.
        """

        feature_series, column_names, index_names = SignCorrelation.check_values(
            feature_series)
        iterations = SignCorrelation.check_iterations(feature_series)

        if not is_return:
            feature_series = SignCorrelation.log_returns(feature_series)

        # Converts the first return table into signs. It will be sliced later
        target_series = np.where(target_series < 0, -1, 1)

        res = np.empty((len(target_series), iterations))
        res[:] = np.nan

        # Fill-in each column based on its length
        for i in range(iterations):
            tmp_feature = np.where(
                feature_series[np.isfinite(feature_series[:, i]), i] < 0, -1, 1)
            index_slice = len(target_series) - len(tmp_feature)
            tmp_target = target_series[index_slice:]
            res[index_slice:, i] = np.cumsum(tmp_target * tmp_feature)

        s_corr = cls(cum_corr=res, column_names=column_names,
                     index_names=index_names)

        fig = s_corr.plot()
        if show_plot:
            fig.show()

        return s_corr
