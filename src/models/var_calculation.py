import numpy as np
import pandas as pd

class VarReturnsCalculator:

  def __init__(self, returns: np.ndarray, time_index: np.ndarray) -> None:

    self.returns = returns
    self.time_index = time_index 
    self.n = len(set(time_index))
  
  @classmethod
  def from_data(cls, returns: np.ndarray, time_index: np.ndarray = None) -> "VarReturnsCalculator":
    """
    Class constructor for the class VarReturnsCalculator

    If no time index is passed, the method will check if returns has an index.
    If yes, it will use this index. If not, it will generate a default index.

    If separate returns and time_index arrays are passed, they must have the
    same order

    Parameters
    ----------
    returns : np.ndarray
        Returns on which to compute the var
    time_index : np.ndarray, optional
        A time value for each observation, by default None
        It is used to weight first observations less than more recent ones

    Returns
    -------
    VarReturnsCalculator
        An instance of the class VarReturnsCalculator
    """

    if isinstance(returns, (list, tuple)):
      returns = np.array(returns)
    elif isinstance(returns, (pd.Series, pd.DataFrame)):
      if time_index is None:
        time_index = returns.index.to_list()
      returns = returns.to_numpy()

    if time_index is None: 
      time_index = np.arange(len(returns))

    vrc = VarReturnsCalculator(returns, time_index)

    return vrc

  def compute_historical_var(self, confidence: float = 0.95) -> float:
    """
    Computes historical unweighted VaR. 

    Parameters
    ----------
    confidence : float, optional
        Confidence level for the VaR, by default 0.95

    Returns
    -------
    Var, float
        The VaR for the confidence level passed
    """

    confidence_inverted = 1 - confidence

    sorted_returns = np.sort(self.returns)
    
    return np.quantile(sorted_returns, confidence_inverted)

  def compute_time_weighted_var(self, confidence: float = 0.95, lam: float = 0.5) -> float:
    """
    Computes historical weighted VaR. Requires a time element to the data. 
    Earlier data are weighted less than more recent data. 

    Weights are calculated with formula :
          wi = lam ** i * (1 - lam) / (1 - lam ** n) for all i = [1 ... n]

    Parameters
    ----------
    confidence : float, optional
        confidence level for the VaR, by default 0.95
    lam : float, optional
        lambda term, used to build the weight, by default 0.5
        Can take any value in the interval ]0, 1]
        Values closer to 1 mean earlier values are less deflated
        Values closer to 0 mean earlier values are more deflated
        
    Returns
    -------
    float
        The time weighted VaR for the confidence level and time index passed
    """

    confidence_inverted = 1 - confidence

    #Application of the formula in the docstring but in vectorised format
    #Creation of a vector of equal values lam of size n, eac
    lam_vec = np.ones(self.n) * lam
    power_vec = np.arange(self.n)
    wts = (np.power(lam_vec, power_vec) 
           * (1 - lam)
           / (1 - lam ** self.n))
    
    time_index = np.unique(self.time_index)

    #Calculate weights and match them so that we can join on rescaled_df on a many to one relation 
    var_df = pd.DataFrame([self.returns], columns=["returns"], index=self.time_index)
    weights_df = pd.DataFrame([wts], columns=["weights"], index=np.sort(time_index)[::-1])

    weighted_df = (var_df.join(weights_df)
                        .sort_values(by='return')
                        .reset_index(drop = True))
    weighted_df['cumulative'] = weighted_df["weights"].cumsum()

    ind = np.argmin(np.abs(weighted_df["cumulative"] - confidence_inverted))

    #Interpolating exact results
    xp = weighted_df.loc[ind: ind + 1, 'cumulative'].values
    fp = weighted_df.loc[ind: ind + 1, 'return'].values

    return np.interp(confidence, xp, fp) 

  def quantile_from_value(self, value : float) -> int:
    """
    Calculates the proportion of observations above a specific value

    Parameters
    ----------
    value : float
        The given value

    Returns
    -------
    int
        The proportion
    """

    prop_above_value = 1 - np.mean(self.returns < value)
    return int(prop_above_value)
