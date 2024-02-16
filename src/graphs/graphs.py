import plotly.graph_objects as go

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

from ..models.stats_utils import kde_estimator, get_summary_statistics
from .color_maker import _color_maker
from .charter import *

def _corr_plot(data: pd.DataFrame, x_values: list):

    fig = go.Figure()
    for i, val in enumerate(data["variable"].unique()):
        tmp_ = data[lambda x: x["variable"] == val]
        fig.add_trace(go.Scatter(x=tmp_.index,
                                 y=tmp_["value"],
                                 mode='lines',
                                 marker_color=np.ravel(FULL_PALETTE)[
                                     i % FULL_PALETTE.shape[0] * FULL_PALETTE.shape[1]],
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


def add_kde(fig: go.Figure, data: np.ndarray, **kwargs) -> go.Figure:
    """
    Adds a KDE plot to an existing go.Figure 

    Parameters
    ----------
    fig : go.Figure
        An existing go.Figure
    data : np.ndarray
        The data from which to compute the KDE 

    Returns
    -------
    go.Figure
    """

    margin = kwargs.get("margin")
    if margin is None:
        margin = 0.05

    x_mins = []
    x_maxs = []
    for trace_data in fig["data"]:
        x_mins.append(min(trace_data.x))
        x_maxs.append(max(trace_data.x))
    x_min = min(x_mins) - margin * min(x_mins)
    x_max = max(x_maxs) - margin * max(x_maxs)

    kde_values = kde_estimator(data, lower_bound=x_min, upper_bound=x_max)

    fig.add_trace(go.Scatter(x=np.sort(data),
                             y=kde_values,
                             mode="markers"))

    return kde_values


def _liner_plot(index: list, values: list, title: str, name: str) -> go.Figure:
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
    trace = go.Scatter(x=index, y=values, mode='lines',
                       line=dict(color=CORAL[0]), name=name)
    layout = dict(title=title,
                  xaxis_title="Time",
                  yaxis_title="Performance",
                  xaxis_tickvals=index[::4],
                  xaxis_ticktext=[str(val) for val in index[::4]]
                  )
    fig = go.Figure(data=[trace], layout=layout)
    return ardian_charter_plotly(fig)


def cum_returns_plotly(pe_fund: pd.Series,
                       public_index: pd.Series,
                       name_portfolio: str = "PE",
                       name_index: str = "Public",
                       timeline: np.ndarray = None,
                       width: int = 1200,
                       heigth: int = 600,
                       **kwargs) -> go.Figure:
    """
    Creates a linechart of cumulative returns for a PE fund and an index. 

    Calculations are done on base 100 with the baseline being the start date of the PE fund

    If the timeline for the graph is not the index of pe_fund, you must pass a vector of date for the index. 

    Parameters
    ----------
    pe_fund : pd.Series
        A series of returns for a pe fund
    public_index : pd.Series
        A series of returns for a public index
    name_portfolio : str
        The name of the PE portfolio, by default PE
    name_index : str
        The name of the public index, by default Public
    timeline : np.ndarray
        The x-axis for the graph, by default none
        Will only be used if pe_fund is not a pd.Series with a valid index

    Returns
    -------
    go.Figure
        A plotly figure comparing returns for a PE fund and a public fund
    """

    if type(pe_fund) == pd.Series:
        timeline = pe_fund.index
    elif (timeline is None):
        raise ValueError(
            "If pe_fund has no index, you must pass an array-like object to timeline to act as the x_axis")

    result_returns = [100]
    for pub in pe_fund:
        result_returns.append(result_returns[-1] * (1 + pub/100))

    base_100_index = 100 * public_index.values/public_index[0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=timeline, y=base_100_index,
                  mode="lines", name=f"{name_index}", marker_color=DARK_GREY[0]))
    fig.add_trace(go.Scatter(x=timeline, y=result_returns,
                  mode="lines", name=f"{name_portfolio}", marker_color=CORAL[1]))
    #fig.add_trace(go.Scatter(x= timeline, y=result_returns_public, name = "Public returns"))
    fig.add_trace(go.Scatter(x=[timeline[0], timeline[len(
        pe_fund)-1]], y=[100, 100], mode="lines", name="Baseline", marker_color=BLUE_GREY[1]))

    if kwargs.get("risk_free_index") is not None:
        twopct_arr = kwargs.get("risk_free_index")
        risk_free_returns = [
            100 * (1 + val/100)**i for i, val in enumerate(twopct_arr)]
        fig.add_trace(go.Scatter(x=timeline, y=risk_free_returns,
                      mode="lines", name="Risk free return", marker_color=DARK_GREY[1]))

    fig.update_layout(xaxis_title="Quarters elapsed",
                      yaxis_title="Cumulative return from baseline",
                      title=f"Comparison of results between {name_portfolio} and {name_index}")

    fig = ardian_charter_plotly(fig, width, heigth)

    return fig

def get_histplot_quantile(series, bins=15, width=12, height=8, legend=False, P99=True):
    P90 = series.quantile(.1)
    P95 = series.quantile(.05, interpolation='lower')
    P99 = series.quantile(.01)
    mean = series.mean()
    fig, ax = ardian_charter_graph(width, height)
    sns.histplot(series, bins=bins, ax=ax, kde=True, color=[
                 BLUE_GREY[1]], stat="density", element="step")
    if P99 == True:
        plt.axvline(P99, color=CORAL[0])
    plt.axvline(P95, color=CORAL[0])
    plt.axvline(mean, color=BLUE_GREY[0])
    plt.axvline(0, color="black")
    # plt.axvline(P90, color=CORAL[2])
    if legend:
        plt.legend(["Production", "P99", "P95", "P90"])
        ax.set_xlabel("", fontsize=20)
        ax.set_ylabel("Probability", fontsize=20)
    return ax

def create_plotly_table(data: np.ndarray,
                        index: list,
                        columns: list,
                        name: str,
                        colors : tuple = ("white", "white"), 
                        threshold: int = 1) -> go.Figure:
    """
    Creates a plotly table with columns

    Parameters
    ----------
    data : np.ndarray
        Array must have as values the Public Market Equivalent for a combination of PE fund and public index
    index : list
        Axis 0 of the table
    columns : list
        Axis 1 of the table 
    name : str
        The name of the table
    colors : array_like/tuple
        The colors to be used with the threshold to colour cells, by default ("white", "white") 
        Only relevant if you are making a table of identical metrics. 
        By default set to ("white", "white") which makes the threshold non-operating. 
    Returns
    -------
    go.Figure
        A plotly table of Public Market Equivalent performances 
    """

    # Used to rotate the table if columns and index are inverted. In this case, table is presented with columns as index and vice versa
    color_index = [DARK_GREY[2] for _ in range(len(index))]
    color_header = [CORAL[1] for _ in range(len(columns))]

    header = dict(values=[name.capitalize()] + columns,
                  fill_color=["white"] + color_header,
                  align='center',
                  font=dict(color='black', size=12),
                  height=30)

    # Creates a list of lists, each column being a list
    cell_values = [list(map(lambda x: np.round(x, 3), data[:, i]))
                   for i in range(data.shape[1])]

    # Creates the same structure, but with colors instead of values
    color_values = [_color_maker(data[:, i], threshold, colors = colors)
                    for i in range(data.shape[1])]
    
    # Concatenate the index column and its colour in the table as the first column
    cell_values = index + cell_values
    color_values = color_index + color_values

    cells = dict(values=cell_values,
                 fill_color=color_values,
                 align='center',
                 font=dict(color='black', size=11),
                 height=25)

    # Create table
    table = go.Figure(data=[go.Table(header=header, cells=cells)])

    # Show the table
    return table

