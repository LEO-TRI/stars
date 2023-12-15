import pandas as pd
import numpy as np
import numpy_financial as npf   
from copy import deepcopy

from .financial_graphs import _liner_plot
from .financial_utils import _find_series_index, _series_to_array

#####FINANCIAL TOOLBOX#####
class FinancialReport:
  """
  Class to compute and hold financial metrics for ONE fund

  All methods use the data passed when the class is instantiated to function

  Has functions for TVPI, MOIC and IRR 
  """

  def __init__(self, contributions: pd.Series, 
               distributions: pd.Series, 
               net_asset_value: pd.Series, 
               discount_rate:float = 0.1, 
               periods: int=4) -> None:
    """
    Initialize an instance of the FinancialReport class.

    Parameters
    ----------
    contributions : array-like, ideally pd.Series
        Series containing contribution values over time.
    distributions : array-like, ideally pd.Series
        Series containing distribution values over time.
    net_asset_value : array-like, ideally pd.Series
        Series containing net asset value values over time.
    discount_rate : float, optional, default: 0.1
        The discount rate used in calculations.

    Returns
    -------
    None
        The function initializes the FinancialReport instance with the provided data.

    Notes
    -----
    - The input Series should represent values over time, and their alignment is important for accurate calculations.
    - The time_index is determined based on the provided contributions and distributions.
    """
    self.contributions = _series_to_array(contributions)
    self.distributions = _series_to_array(distributions)
    self.net_cashflows = self.distributions - self.contributions
    self.discount_rate = discount_rate
    self.periods = periods
    
    #Keeps in memory the original data index if at least one argument has an index, else takes a default index 
    self.time_index = _find_series_index(contributions, distributions) 
    self.net_asset_value = _series_to_array(net_asset_value) 
    
  def compute_all_ratios(self) -> pd.DataFrame:
    """
    Computes all ratio methods and store them in a pd.DataFrame

    Returns
    -------
    value_dict :  pd.DataFrame
        A DataFrame of metrics ratios
    """

    self.value_dict = dict(dpi=self.compute_dpi(),
                            rvpi=self.compute_rvpi(),
                            tvpi=self.compute_tvpi(),
                            moic=self.compute_moic(),
                            netCumCashflows=np.cumsum(self.net_cashflows),
                            irr=self.compute_irr(),
                            )
        
    return pd.DataFrame(self.value_dict)


  def compute_dpi(self) -> np.ndarray:
      """
      Compute the Distributed to Paid-In (DPI) ratio.

      Returns:
      - numpy.ndarray: Array of DPI values at each time period.
      """
      return np.cumsum(self.distributions) / np.cumsum(self.contributions)

  def compute_rvpi(self) -> np.ndarray:
      """
      Compute the Residual Value to Paid-In (RVPI) ratio.

      Returns:
      - numpy.ndarray: Array of RVPI values at each time period.
      """
      return self.net_asset_value / np.cumsum(self.contributions)

  def compute_tvpi(self) -> np.ndarray:
      """
      Compute the Total Value to Paid-In (TVPI) ratio by summing the DPI and RVPI.

      Returns:
      - numpy.ndarray: Array of TVPI values at each time period.
      """
      return self.compute_dpi() + self.compute_rvpi()

  def compute_moic(self) -> np.ndarray:
      """
      Compute the Multiple on Invested Capital (MOIC) ratio. 
      Same as tvpi at period T.

      Returns:
      - numpy.ndarray: Array of MOIC values at each time period.
      """
      return ((self.net_asset_value + np.cumsum(self.distributions)) 
                / np.sum(self.contributions)
                )
  
  def compute_irr(self) -> float:
    """
    Compute the Internal Rate of Return (IRR) for the cashflows and NAV.

    Returns:
    - float: The computed IRR.

    """
    #IRR at year 0 is 0 and we then append computed results
    irr_list = [0]
    n_steps = len(self.net_cashflows)
    
    if n_steps == 1:
       return np.asarray(irr_list)
    
    else:
        #2 potential options to calculate IRR: adding the NAV to the last value or not
        #For the time being, we use the NAV to be coherent with the previous works we did. 
        #But this strongly inflates early IRR values

        #Option 1 with the NAV:
        for i in range(1, n_steps): 
            #We use i + 1 since slices are exclusive for the upper bound
            rescaled_cashflows = deepcopy(self.net_cashflows[:i+1])
            rescaled_cashflows[i] += self.net_asset_value[i]
            irr_list.append(npf.irr(rescaled_cashflows))
        
        #Option 2 without the NAV:
        #irr_list = irr_list + [npf.irr(self.net_cashflows[:i+1]) for i in range(1, n_steps)]
    
    irr_array =  np.asarray(irr_list)

    return (1 + irr_array) ** self.periods - 1

  def plot(self, 
           data, 
           title: str = None, 
           name: str = None):
    """
    Plot the specified data using a plot. Way to integrate a plotting function into the class. 

    Parameters:
    ----------
    - data: Either a string specifying a precomputed metric or a numpy array representing the data.
    - title (str): The title of the plot.
    - name (str): The name of the plot (used in legends).

    Returns:
    ----------
    - plotly.graph_objs.Figure: A Plotly figure object.
    """
    if isinstance(data, str):
        values = self.value_dict.get(data)
        if values is None:
            raise ValueError("The metric you are trying to plot has not been computed")
        if title is None:
            title = f"{data} across time"
            name=data
        return _liner_plot(self.time_index, values=values, title=title, name=name)

    return _liner_plot(self.time_index, values=data, title=title)
  
