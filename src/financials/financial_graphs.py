import plotly.graph_objects as go
import pandas as pd
import numpy as np


###Contains graphs for plotting correlation and returns 
# color constants and palettes
CORAL = ["#D94535", "#ed8a84", "#f3b1ad", "#f9d8d6"]
DARK_GREY = ["#323C46", "#77949d", "#a4b7be", "#d2dbde"]
STONE = ["#7d8791", "#a5aaaa", "#d2d2d2", "#ebebeb"]
BLUE_GREY = ["#94b7bb", "#bfd4d6", "#d4e2e4", "#eaf1f1"]
BROWN = ["#7a6855", "#b4a594", "#cdc3b8", "#e6e1db"]
PURPLE = ["#8d89a5", "#bbb8c9", "#d1d0db", "#e8e7ed"]

# Artificial extension of the palette table for repetition
FULL_PALETTE = np.vstack([CORAL, DARK_GREY, STONE, BLUE_GREY, BROWN, PURPLE])

def _color_maker(table_col: np.ndarray,
                 threshold: float = 0,
                 colors: list = None,
                 is_color_smaller: bool = True) -> np.ndarray:
    """
    Colours a sequence based on an inequality and a threshold.

    Arguments will determine the values  

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

    dark_grey = DARK_GREY[1]
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


def _create_plotly_table(data: np.ndarray, index: list, columns: list, threshold: int, metric: str) -> go.Figure:
    """
    Creates a plotly table of PE fund's Public Market Equivalent

      Parameters
      ----------
      data : np.ndarray
          Array must have as values the Public Market Equivalent for a combination of PE fund and public index
      index : list
          Axis 0 of the table
      columns : list
          Axis 1 of the table 
      metric : str, optional
          The metric in the table, by default "kspme"
          Can be one of ('direct_alpha', 'kspme')

      Returns
      -------
      go.Figure
          A plotly table of Public Market Equivalent performances 
    """

    # Used to rotate the table if columns and index are inverted. In this case, table is presented with columns as index and vice versa
    if len(columns) < data.shape[1]:
        data = data.T
        color_index = [CORAL[1] for _ in range(len(index))]
        color_header = [DARK_GREY[2] for _ in range(len(columns))]

    else:
        color_index = [DARK_GREY[2] for _ in range(len(index))]
        color_header = [CORAL[1] for _ in range(len(columns))]

    header = dict(values=[metric.capitalize()] + columns,
                  fill_color=["white"] + color_header,
                  align='center',
                  font=dict(color='black', size=12),
                  height=30)

    # Creates a list of lists, each column being a list
    cell_values = [list(map(lambda x: np.round(x, 3), data[:, i]))
                   for i in range(data.shape[1])]

    # Creates the same structure, but with colors instead of values
    color_values = [_color_maker(data[:, i], threshold)
                    for i in range(data.shape[1])]

    cell_values.insert(0, index)  # Insert the index column and its colour
    color_values.insert(0, color_index)

    cells = dict(values=cell_values,
                 fill_color=color_values,
                 align='center',
                 font=dict(color='black', size=11),
                 height=25)

    # Create table
    table = go.Figure(data=[go.Table(header=header, cells=cells)])

    # Show the table
    return table



def _corr_plot(data: pd.DataFrame, x_values: list, color_palette: np.ndarray = FULL_PALETTE) -> go.Figure:
    """
    Generate a correlation plot using Plotly.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the data.
    x_values : list
        The list of column names to use as x-axis values.
    color_palette : np.ndarray
        The color palette, a 2d array with rows being colours and columns being shades

    Returns
    -------
    go.Figure
        A Plotly figure object representing the correlation plot.
    """

    fig = go.Figure()
    for i, val in enumerate(data["variable"].unique()):
        tmp_ = data[lambda x: x["variable"] == val]
        fig.add_trace(go.Scatter(x=tmp_.index,
                                 y=tmp_["value"],
                                 mode='lines',
                                 marker_color=np.ravel(color_palette)[
                                     i % color_palette.shape[0] * color_palette.shape[1]],
                                 name=val
                                 ),
                      )

    fig.add_trace(go.Scatter(x=[data.index[0], data.index[-1]],
                             y=[0, 0],
                             mode='lines',
                             line=dict(dash='dash'),
                             marker_color=FULL_PALETTE[1, 1],
                             name='Correlation = 0'))

    fig.update_layout(xaxis=dict(tickmode='array',
                                 tickvals=x_values[::5],
                                 ticktext=x_values[::5],
                                 title="Date"
                                 ),
                      yaxis_title="Correlation"
                      )

    return fig


def _liner_plot(index: list, values: list, title: str, name: str, color: str = "#D94535") -> go.Figure:
    """
    Generate a line plot with x as the index and y as the values.

    Parameters
    ----------
    index : array_like
        The x-axis values.
    values : array_like
        The y-axis values.
    title : str
        The title of the plot.
    name : str
        The name of the plot.
    color: str
         The color for the line plot, by default "#D94535"

    Returns
    -------
    go.Figure
        A Plotly figure object representing the line plot.
    """
    trace = go.Scatter(x=index, 
                       y=values, 
                       mode='lines',
                       line=dict(color=color), 
                       name=name)
    
    layout = dict(title=title,
                  xaxis_title="Time",
                  yaxis_title="Performance",
                  xaxis_tickvals=index[::4],
                  xaxis_ticktext=[str(val) for val in index[::4]]
                  )

    fig = go.Figure(data=[trace], layout=layout)

    return fig
