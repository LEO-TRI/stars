import plotly.graph_objects as go
import numpy as np
import pandas as pd
from numbers import Number

CORAL = ["#D94535", "#ed8a84", "#f3b1ad", "#f9d8d6"]
DARK_GREY = ["#323C46", "#77949d", "#a4b7be", "#d2dbde"]
STONE = ["#7d8791", "#a5aaaa", "#d2d2d2", "#ebebeb"]
BLUE_GREY = ["#94b7bb", "#bfd4d6", "#d4e2e4", "#eaf1f1"]
BROWN = ["#7a6855", "#b4a594", "#cdc3b8", "#e6e1db"]
PURPLE = ["#8d89a5", "#bbb8c9", "#d1d0db", "#e8e7ed"]
PALETTE = CORAL + BLUE_GREY[::-1]

FULL_PALETTE = np.asarray([CORAL, DARK_GREY, BLUE_GREY, STONE, BROWN, PURPLE])

class FlowChart:
    
    def __init__(self, data : np.ndarray, x_axis : np.ndarray, cols: list[str] = None, fig: go.Figure = None) -> None:
        self.data = data
        self.x_axis = x_axis
        self.keys = cols
        
        self.fig = fig

    @classmethod
    def from_wide_data(cls, 
                       data : np.ndarray, 
                       x_axis: np.ndarray = None, 
                       cols: list[str] = None, 
                       title : str = "", 
                       graph_kwargs: dict = {}):
        
        if cols is None:
            cols = data.columns.to_list()

        if x_axis is None:
            if isinstance(data, (pd.DataFrame, pd.Series)):
                x_axis = data.index 
            else:
                x_axis = np.arange(len(data))

        data_dict = data[cols].select_dtypes(np.number).to_dict("list")

        return FlowChart(data_dict, x_axis).make_flow_chart(data_dict, title, graph_kwargs)

    @classmethod
    def from_long_data(cls, 
                       data : pd.DataFrame, 
                       col_keys : str, 
                       col_val : str, 
                       x_axis: np.ndarray = None, 
                       title : str = "", 
                       graph_kwargs: dict = {}):
        
        unique_keys = list(data[col_keys].unique())
        
        data = pd.pivot(data, columns = col_keys, values = col_val)
        
        return cls.from_wide_data(data, x_axis, unique_keys, title, graph_kwargs)
        
    def make_flow_chart(self, 
                        data_dict : dict, 
                        title: str = "", 
                        graph_kwargs: dict = {}) -> go.Figure:
        
        if isinstance(data_dict, (pd.DataFrame)):
            data_dict = data_dict.select_dtypes(np.number).to_dict("list")
        
        graphs_list = []
    
        for i, (key, val) in enumerate(data_dict.items()):
            
            if len(val) < len(self.x_axis):
                size = len(val) - len(self.x_axis)
                size = np.zeros(size)
                if graph_kwargs.get("padding_back", False):
                    val = np.concatenate([val, size])
                else: 
                    val = np.concatenate([size, val])
            
            fig_dict = go.Scatter(x = self.x_axis,
                       y = val,
                       name = key,
                       stackgroup='one',
                       mode='lines',
                       line=dict(width=0.5, color=FULL_PALETTE[i,0]),
                       )
            
            graphs_list.append(fig_dict)
            
        layout_dict = dict(title = dict(text=title),
                           yaxis = dict(range=[0, 1]))
            
        fig = go.Figure(graphs_list, layout_dict)

        return fig
