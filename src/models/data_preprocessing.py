import pandas as pd
import numpy as np
import warnings

###Returners###
def no_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices

def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    '''
    Input:
    prices (df) - Dataframe of close prices
    ma (int) - legth of the moveing average
    
    Output:
    prices(df) - An enhanced prices dataframe, with moving averages and log return columns
    prices_array(nd.array) - an array of log returns
    '''

    prices = prices.select_dtypes(include = np.number).sort_index()
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log_prices = np.log(prices.to_numpy()/prices.shift(1).to_numpy())

    return pd.DataFrame(log_prices, index = prices.index, columns = prices.columns).dropna(how = "all").fillna(0)

def normal_returns(prices: pd.DataFrame) -> pd.DataFrame:
    '''
    Input:
    prices (df) - Dataframe of close prices
    ma (int) - legth of the moveing average
    
    Output:
    prices(df) - An enhanced prices dataframe, with moving averages and log return columns
    prices_array(nd.array) - an array of log returns
    '''

    prices = prices.select_dtypes(include = np.number).sort_index()
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        normal_prices = (prices.to_numpy() - prices.shift(1).to_numpy()) / prices.shift(1).to_numpy()
    
    return pd.DataFrame(normal_prices, index = prices.index, columns = prices.columns).dropna(how = "all").fillna(0)


###Smoothers####
def roller_smoother(prices: pd.DataFrame, alpha: float = 0.25, window: int = 5) -> np.ndarray:
    '''
    Input:
    prices (df) - Dataframe of close prices
    ma (int) - legth of the moveing average
    
    Output:
    prices(df) - An enhanced prices dataframe, with moving averages and log return columns
    prices_array(nd.array) - an array of log returns
    '''

    prices = prices.select_dtypes(include = np.number).sort_index()
    
    prices_rolling = prices.rolling(window, center = True).mean()

    return prices_rolling

def exponential_smoother(prices : pd.DataFrame, alpha: float = 0.25, window: int = 5):
    
    prices_arr = prices.to_numpy()
    prices_ema = [prices_arr[0,:]]

    for i in range(1, prices_arr.shape[0]):
        transformed_arr = alpha * prices_arr[i,:] + (1 - alpha) * prices_ema[-1]
        prices_ema.append(transformed_arr)

    return pd.DataFrame(np.array(prices_ema), index = prices.index, columns = prices.columns)


###Standardisers####
def window_standardiser(prices : pd.DataFrame, window : int = 30):
    
    std_rolled = prices.rolling(window).std().to_numpy()
    mean_rolled = prices.rolling(window).mean().to_numpy()    
    
    price_standardised = (prices.to_numpy() - mean_rolled) / std_rolled

    return (pd.DataFrame(price_standardised, index = prices.index, columns = prices.columns)
              .dropna(how = "all")
              .fillna(0))

def normal_standardiser(prices : pd.DataFrame, window : int = 30):
    
    std_sample = np.std(prices, axis = 0)
    mean_sample = np.mean(prices, axis = 0)

    price_standardised = ((prices - mean_sample) / std_sample).to_numpy()

    return (pd.DataFrame(price_standardised, index = prices.index, columns = prices.columns)
              .dropna(how = "all")
              .fillna(0))
