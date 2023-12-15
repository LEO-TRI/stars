import numpy as np
import pandas as pd 
import numbers 

#Contains a series of functions used inside scripts for data manipulation e.g. check types of objects, and transform them

def _find_series_index(*args):
  """
  Find the index of the first pandas Series in a variable-length argument list.

  Parameters
  ----------
  *args : pandas.Series or array-like
      Variable-length argument list containing data.

  Returns
  -------
  list
      A list representing the index of the first pandas Series found in the argument list, or a default list of values going from 0 to len(arg1) -1 .

  """

  for arg in args:
      if isinstance(arg, pd.Series) | isinstance(arg, pd.DataFrame) :
          return arg.index.to_list()
        
  return list(range(len(args[0])))

def _is_array_like(obj) -> bool:
  """
  Check if an object is array_like, returns True if yes, False else

  Array_like is defined as either list, numpy.array, panda.Series

  Used in functions to determine the type of passed arguments

  """
  return any((isinstance(obj, list), 
              isinstance(obj, np.ndarray), 
              isinstance(obj, pd.Series))
              )

def _is_numeric(obj) -> bool:
  """
  Check if an object is a number and returns True if yes, False else
  """
  return isinstance(obj, numbers.Number)

def _series_to_array(obj) -> np.ndarray:
    """
    Converts an array like object into an array 

    Parameters
    ----------
    obj:
      Any array_like object
    
    Returns
    -------
    np.ndarray
    """
    if isinstance(obj, pd.Series) | isinstance(obj, pd.DataFrame): 
      return obj.to_numpy()
    elif isinstance(obj, list):
      return np.array(obj)
    return obj

def _expand_to_2d(arr: np.ndarray) -> np.ndarray:
  """
  Check the number of dimensions of a NumPy array and expand it to 2 dimensions if necessary.

  Parameters:
  - arr (numpy array): Input array.

  Returns:
  - numpy array: The input array, expanded to 2 dimensions if it was originally 1-dimensional.
  """
  if np.ndim(arr) == 1:
      # If the array has only one dimension, expand it to 2 dimensions
      return np.expand_dims(arr, axis=1)
  else:
      # If the array already has more than one dimension, return it as is
      return arr

def _reduce_to_1d(arr: np.ndarray) -> np.ndarray:
  """
  Check the size of axis 1 of a NumPy array and reduce it to 1 dimension if the size is 1.

  Parameters:
  - arr (numpy array): Input array.

  Returns:
  - numpy array: The input array, reduced to 1 dimension if the size of axis 1 is 1.
  """
  if arr.shape[1] == 1:
      # If the size of axis 1 is 1, reduce the array to 1 dimension
      return np.squeeze(arr, axis=1)
  else:
      # If the size of axis 1 is greater than 1, return it as is
      return arr
