import numpy as np
from sklearn.linear_model import LinearRegression

def rolling_window(series: np.ndarray, step: int, axis: int = 0) -> list:
    """
    Slice a dataframe into several windows based on a specific step size. 

    Parameters
    ----------
    series : numpy.ndarray
        The input array for which a rolling window will be created. The input array should be a numpy array, list of lists or dataframe
    step : int
        The size of the rolling window.
    axis : int, optional, default: 0
        The axis along which the rolling window will be applied.

    Returns
    -------
    list
        A list containing views of the input array with a rolling window along the specified axis.
    """

    dims = list(series.shape) + [series.shape[axis] - step]
    dims[axis] = step

    results = []  # np.zeros(dims, dtype=int)

    slicing_index = [slice(None)] * len(series.shape)

    series = np.array(series)

    # Generates a new slicing object at each iteration to move the window and uses it to slice the dataframe
    for i in range(series.shape[axis] - step):
        slicing_index[axis] = slice(i, i + step)
        results.append(series[tuple(slicing_index)])

    return results


def rolling_regression(X: np.ndarray, y: np.ndarray, window_size: int, fit_intercept: bool=False) -> np.ndarray:
    """
    Performs a rolling linear regression and returns coefs and intercept based on a for loop and refiting the model 

      Parameters
      ----------
      X : array_like
          The independent variables
      y : array_like
          The dependent variable
      fit_intercept : bool, optional
          Whether to fit the intercept, by default False

      Returns
      -------
      tuple
          The models coefs and intercept (if fit_intercept = True)
    """

    X = np.array(X) 
    y = np.array(y)
  
    results = []
    for i in range(len(y) - window_size):
      X_rolling = X[i:i+window_size, ...] #Ellipsis in case the array has more than 1 dimension
      y_rolling = y[i:i+window_size] 

      model = LinearRegression(fit_intercept=fit_intercept).fit(X_rolling, y_rolling)
      results.append(np.concatenate([model.coef_.ravel(), model.intercept_]))
    
    return np.array(results)

