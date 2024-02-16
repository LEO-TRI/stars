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

class Marimekko:
    
    def __init__(self, data, width: int, labels: str = None, cols : str = None) -> None:
        self.data = data
        self.width = width
        self.labels = labels
        self.keys = cols

    @classmethod
    def from_wide_data(cls, data, width: int, labels: str = None, cols: str = None, title : str = ""):
        
        if cols is None:
            cols = data.columns.to_list()


        if labels is None:
            labels = data.index.to_list()

        data = data[cols].select_dtypes(np.number)

        data_dict = data.to_dict("list")

        return Marimekko(data_dict, width, labels, cols).make_meko_chart(data_dict, width, labels, title)

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
        
    def make_meko_chart(self, data_dict : dict, width: int, labels: str = "None", title: str = ""):

        data_len = len(data_dict.get(self.keys[0]))
        if isinstance(width, Number):
            width = np.zeros(data_len) + width

        fig = go.Figure()
    
        for i, key in enumerate(data_dict.keys()):
            fig.add_trace(
                go.Bar(
                    name=key,
                    marker_color=FULL_PALETTE[i,0],
                    y=data_dict[key],
                    x=np.cumsum(width) - width,
                    width=width,
                    offset=0,
                    customdata=np.transpose([labels, width * data_dict[key]]),
                    texttemplate="%{y} x %{width} =<br>%{customdata[1]}",
                    textposition="inside",
                    textangle=0,
                    textfont_color="white",

                ),
            )

        filter_condition = lambda date: (date.month == 12 and date.day == 31) # or (date.month == 6 and date.day == 30)
        filtered_dates = [(index, str(date.year)) for index, date in enumerate(labels) if filter_condition(date)]
    
        filtered_dates = pd.DataFrame(filtered_dates, columns = ["ind", "date"])

        fig.update_yaxes(range=[0, 1])

        labels = [str(val.month) + "-" + str(val.year) for val in labels]

        fig.update_layout(
            xaxis = dict(tickmode = 'array',
                        tickvals = list(range(len(labels)))[7::12],
                        ticktext = labels[7::12]),
            title_text=title,
            barmode="stack",
            uniformtext=dict(mode="hide", minsize=10),
        )

        return fig
