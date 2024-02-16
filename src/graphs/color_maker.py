import numpy as np
import pandas as pd

DARK_GREY = ["#323C46", "#77949d", "#a4b7be", "#d2dbde"]


def _color_maker(table_col: np.ndarray,
                 threshold: float = 0,
                 colors: list = None,
                 is_color_smaller: bool = True) -> np.ndarray:
    """
    Colours a sequence based on an inequality and a threshold 

    Parameters
    ----------
    table_col: np.ndarray
      A 1D np.array, tuple, list or pd.Series. Values will determine the colours based on a threshold
    threshold: float 
      Threshold to determine on which side of the inequality values will fall, by default 0 
    colors: array_like
      An array of colours to be used to colour cells, use default set if None is passed, by default None.
      Colour that should be returned when the inequality is true should alaways be passed first 
    is_color_smaller: bool
      Whether to colour cells that are above or below the threshold, by default True meaning colouring cells below the threshold

    Returns
    -------
    np.ndarray
        An array of colours based on the value of table_col

    """

    dark_grey = DARK_GREY[0]
    if colors is None:
        small_colour, large_colour = (
            dark_grey, "white") if is_color_smaller else ("white", dark_grey)

    elif isinstance(colors, (list, tuple, np.ndarray, pd.Series)):
        small_colour, large_colour = colors if is_color_smaller else colors[::-1]

    elif isinstance(colors, str):
        small_colour, large_colour = (
            colors, "white") if is_color_smaller else ("white", colors)

    else:
        raise TypeError("Colour must be a string or a list/tuple of colours")

    return list(np.where(table_col < threshold, small_colour, large_colour))
