import pandas as pd
import numpy as np
from .financial_utils import _expand_to_2d, _reduce_to_1d
import warnings

class ShiftReturnsMixin:
    """
    Mixin

    Class carrying functions to obtain normal and log returns from absolute values

    Mixed in when necessary with other finance classes
    """

    def __init__(self):
        pass

    @staticmethod
    def log_returns(values: np.ndarray) -> np.ndarray:
        """
        Divide an array by its shifted version and take the log

        Parameters:
        - data: numpy array or pandas dataframe

        Returns:
        - numpy.ndarray: Array of log returns.
        """
        if isinstance(values, pd.DataFrame):
            values = values.to_numpy()

        values = _expand_to_2d(values)
        # Shift by 1 the array, divide and concat with an additional nan layer, then take log returns

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            values = np.log(np.vstack([np.empty(values.shape[1]) * np.nan,
                                    (values[1:] / values[:-1]),
                                    ]
                                    )
                            )

        return _reduce_to_1d(values)

    @staticmethod
    def normal_returns(values: np.ndarray) -> np.ndarray:
        """
        Calculate normal returns from a 1D or 2D array of values.

        Parameters:
        - values (numpy.ndarray or pandas.DataFrame): Array of values representing the asset prices or returns.

        Returns:
        - numpy.ndarray: Array of normal returns. For a 2D input, returns are calculated row-wise.

        Notes:
        - If the input is a pandas DataFrame, it is converted to a NumPy array.
        - The function first ensures that the input is a 2D array using _expand_to_2d.
        - Normal returns are calculated as the percentage change in values.
        - NaN is inserted as the first element of each row to maintain the original array shape.
        - The resulting array is reduced to 1D using _reduce_to_1d.
        """
        if isinstance(values, pd.DataFrame):
            values = values.to_numpy()

        values = _expand_to_2d(values)

        returns = np.vstack(
            [
                np.empty(values.shape[1]) * np.nan,
                ((values[1:] - values[:-1]) / values[:-1]),
            ]
        )

        return _reduce_to_1d(returns)
