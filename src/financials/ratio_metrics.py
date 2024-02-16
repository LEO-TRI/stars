import pandas as pd
import numpy as np
from .financial_utils import _is_numeric, _is_array_like, _series_to_array
from .returns_mixin import ShiftReturnsMixin


class RatioMetrics(ShiftReturnsMixin):  
    # TODO check how to harmonise sizes + annualisation
    """
    Compute and holds ratios for a fund.

    Implemented for the time being are rations in annualized version

    Should be built using the from_data class method
    """

    def __init__(
        self,
        returns: np.ndarray,
        index_fund: np.ndarray = None,
        risk_free_rate: float = 0.005,
        periods: int = 4) -> "RatioMetrics":
        """
        Initialize an instance of the RatioMetrics class.

        Parameters
        ----------
        returns : numpy.ndarray
            Array containing returns of the investment over time.
        index_fund : numpy.ndarray
            Array containing values of the index fund (e.g., CAC40) over time, by default None.
        risk_free_rate : numpy.ndarray or float, optional, default: 0.005
            Array containing risk-free rates quarterly.
        periods: int
            Integer indicating how many periods of the data there is in one year e.g. for quarterly data periods=4, by default 4

        Returns
        -------
        None
            The function initializes the RatioMetrics instance with the provided data.
        """

        self.returns = returns
        self.index_fund = index_fund
        self.risk_free_rate = risk_free_rate
        self.periods = periods
        # Placeholder that will be overwritten when calling self.compute_all_ratios
        self.value_dict = None  

    @classmethod
    def from_data(
        cls,
        returns: np.ndarray,
        index_fund: np.ndarray = None,
        risk_free_rate: float = 0.005,
        is_return: bool = False,
        periods: int = 4) -> "RatioMetrics":
        """
        A constructor to build the class.

        Used from processing data and avoid loading too much info in __init__

        Parameters
        ----------
        returns : numpy.ndarray
            Array containing returns of the investment over time.
        index_fund : numpy.ndarray
            Array containing values of the index fund (e.g., CAC40) over time, by default None.
        is_return: bool
            Whether the index_fund variable are returns or values, by default False
        risk_free_rate : numpy.ndarray or float, optional, default: 0.005
            Array containing risk-free rates quarterly.
        periods: int
            Integer indicating how many periods of the data there is in one year e.g. for quarterly data periods=4, by default 4


        Returns
        -------
        RatioMetrics
            An instance of the class RatioMetrics

        Raises
        ------
        ValueError
            Will be raised if risk free rate is array like and not the same length as returns
        ValueError
            Will be raised if risk free rate is not array like or a number

        Notes
        -----
        - The input arrays should represent values over time, and their alignment is important for accurate calculations.
        - If is_return is False, then returns will be calculated. This means all metrics are evaluated on arrays of size n - 1,
          since taking returns mean we lose 1 observation.
        - The RatioMetrics instance is initialized with the provided index_fund, returns, and risk_free_rate arrays.
        - If you have yearly data or don't want data to be annualised, set periods to 1
        """

        index_fund = _series_to_array(index_fund) if index_fund is not None else None
        returns = _series_to_array(returns)

        if _is_numeric(risk_free_rate):
            risk_free_rate = risk_free_rate

        elif _is_array_like(risk_free_rate):
            risk_free_rate = _series_to_array(risk_free_rate)
            if len(returns) != len(risk_free_rate):
                raise ValueError("Risk free rate and returns must be the same length")

        else:
            raise ValueError("Risk free rate must be number or array_like")

        if (not is_return) & (index_fund is not None):
            index_fund = RatioMetrics.normal_returns(index_fund)
            mask = np.isfinite(index_fund)
            index_fund = index_fund[mask]

            returns = returns[mask]
            if isinstance(risk_free_rate, np.ndarray):
                risk_free_rate = risk_free_rate[mask]

        return RatioMetrics(returns, index_fund, risk_free_rate, periods)

    def compute_all_ratios(self) -> dict:
        """
        Computes all ratio methods and store them in a dictionnary

        Returns
        -------
        value_dict :  dict
            A dictionnary of metrics ratios
        """

        self.value_dict = dict(sharpe_ratio=self.compute_sharpe_ratio(),
                          sortino_ratio=self.compute_sortino_ratio(),
                          )
        
        # Add additional metrics that require index funds
        if self.index_fund is not None:  
            self.value_dict.update(dict(modigliani_ratio=self.compute_modigliani_ratio(),
                                   treynor_ratio=self.compute_treynor_ratio(),
                                   jensen_alpha=self.compute_jensen_alpha(),
                                    )
                                   )

        return self.value_dict

    def compute_sharpe_ratio(self) -> float:
        """
        Calculate the Sharpe ratio.

        Parameters
        ----------
        risk_free_rate : np.ndarray or float
            Corresponding risk-free rates for each period. Can be passed as a float or an array when instantiating the class

        Returns
        -------
        float
            sharpe_ratio: The calculated Sharpe ratio.
        """

        excess_returns = self.returns - self.risk_free_rate
        sharpe_ratio_period = np.mean(excess_returns) / np.std(excess_returns)

        return sharpe_ratio_period * (self.periods ** 0.5)

    def compute_fund_beta(self) -> float:
        """
        Calculate the beta of the fund with respect to the index.

        Not annualized by default.

        Returns
        -------
        float
            The beta of the fund relative to the index.

        Notes
        -----
        - The formula used is Covariance(fund_returns, index_returns) / Variance(index_returns).
        """

        covariance_beta = np.cov(self.returns, self.index_fund, rowvar=False)[1, 0]
        return covariance_beta / np.var(self.index_fund)
                

    def compute_treynor_ratio(self) -> float:
        """
        Calculate the Treynor ratio of the investment.

        Returns
        -------
        float
            The Treynor ratio of the investment.

        Notes
        -----
        - The Treynor ratio is computed as (Mean(returns) - Mean(risk_free_rate)) / Beta.
        - Beta is calculated using the compute_fund_beta method.
        """

        excess_returns = np.mean(self.returns) - np.mean(self.risk_free_rate)
        treynor_ratio_period = excess_returns / self.compute_fund_beta()

        return treynor_ratio_period * (self.periods ** 0.5)
    
    def compute_jensen_alpha(self) -> float:
        """
        Calculate Jensen's Alpha of the investment.

        Returns
        -------
        float
            Jensen's Alpha of the investment.

        Notes
        -----
        - Jensen's Alpha is computed as Mean(returns) - Mean(risk_free_rate) - Beta * (Mean(index_returns) - Mean(risk_free_rate)).
        - Beta is calculated using the compute_fund_beta method.
        """

        # index_returns = self.normal_returns(self.index_fund)[1:]
        market_excess_returns = np.mean(self.index_fund) - np.mean(self.risk_free_rate)
        jensen_alpha_period = (np.mean(self.returns) 
                               - np.mean(self.risk_free_rate)
                               - self.compute_fund_beta() * market_excess_returns
                               )
                               
        #(1 + jensen_alpha_period)**self.periods - 1
        return jensen_alpha_period * (self.periods ** 0.5)

    def compute_modigliani_ratio(self) -> float:
        """
        Calculate the Modigliani adjusted risk performance measure.

        Parameters
        ----------
        risk_free_rate : np.ndarray or float
            Corresponding risk-free rates for each period. Can be passed as a float or an array when instantiating the class
        index_fund : np.ndarray
            Corresponding values for a public index for each period

        Returns
        -------
        float
            modigliani_ratio: The calculated Modigliani ratio.
        """

        benchmark_excess_return_std = np.std(self.index_fund - self.risk_free_rate)
        mean_risk_free = np.mean(self.risk_free_rate)

        return (self.compute_sharpe_ratio() 
                 * benchmark_excess_return_std * (self.periods ** 0.5)
                 + mean_risk_free * self.periods
                 ) 
                

    def compute_downside_deviation(self) -> float:
        """
        Calculate the downside deviation of the investment.

        Returns
        -------
        float
            The downside deviation of the investment.

        Notes
        -----
        - MAR is the mean of the risk-free rate.
        - The rescaled returns are calculated by subtracting MAR from the actual returns.
        - The downside deviation is the square root of the sum of squared negative rescaled returns divided by the number of returns.
        """

        mar = np.mean(self.risk_free_rate)
        rescaled_returns = self.returns - mar
        negative_squared = np.where(rescaled_returns < 0, rescaled_returns, 0) ** 2

        return np.mean(negative_squared) ** 0.5

    def compute_sortino_ratio(self) -> float:
        """
        Calculate the yearly Sortino ratio of the investment.

        Returns
        -------
        float
            The Sortino ratio of the investment.

        Notes
        -----
        - The Sortino ratio is computed as (Mean(returns) - Mean(risk_free_rate)) / Downside Deviation.
        - Downside Deviation is calculated using the compute_downside_deviation method.
        """

        mean_excess_returns = np.mean(self.returns) - np.mean(self.risk_free_rate)
        sortino_ratio_period = mean_excess_returns / self.compute_downside_deviation()
        return sortino_ratio_period * (self.periods ** 0.5)
