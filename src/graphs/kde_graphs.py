import numpy as np
import plotly.graph_objects as go
from scipy import stats 

from ..utils.data_utils import _is_array_like

class KdeFactory:
    """
    Recoding of Plotly's figure_factory to get a decent kde 

    Should always be instantiated with the class constructor make_kde()
    """
    
    def __init__(self, hist_data, graph_data, start, end, bin_size=1, colors=None, title: str=None) -> None:
      self.hist_data = hist_data
      self.graph_data = graph_data
      self.colors = colors
      self.start = start
      self.end = end
      self.bin_size=bin_size
      self.title=title

      if (type(self.bin_size) == np.number) & (len(hist_data)>1):
        self.bin_size=np.ones(len(hist_data)) * bin_size

    def plot(self,
             show_hist : bool = True,
             show_curve: bool = True,
             layout_kwargs: dict = None,
             title: str = None) -> "KdeFactory":
      """
      Plots a kde based on the passed data in the class constructor

      Parameters
      ----------
      show_hist : bool, optional 
          Whether to show the histogram, by default True
      show_curve : bool, optional 
          Whether to show the KDE, by default True
      layout_kwargs : dict, optional
          Additional arguments for the layout of the graph, by default None
      title : str, optional
          _description_, by default None

      Returns
      -------
      KdeFactory
          An instance of the class with a plot attribute
      """

      layout = dict(
        barmode="overlay",
        hovermode="closest",
        legend=dict(traceorder="reversed"),
        xaxis=dict(domain=[0.0, 1], anchor="y2", zeroline=False),
        yaxis=dict(domain=[0.0, 1], anchor="free", position=0.0),
        )
      
      if title is not None: 
        layout["title"] = title

      if layout_kwargs is not None: 
        layout.update(layout_kwargs)

      plots_dict = self.graph_data
      
      if show_hist:
        for index, data in enumerate(self.hist_data): 
          hist_plot = dict(type="histogram",
                            x=data,
                            name="Returns",
                            histnorm='probability density',
                            legendgroup=index,
                            showlegend=False,
                            opacity=0.7,
                            xbins=dict(start=self.start[index], 
                                       end=self.end[index],
                                       size=self.bin_size),
                            autobinx=False,
                            marker=dict(color=self.colors[index % len(self.colors)])
                            )
          
          plots_dict.insert(0, hist_plot)

      fig = go.Figure(data=plots_dict, layout=layout)

      self.fig = fig

      if show_curve:
        fig.show()

      return self

    @classmethod
    def make_kde(cls,
                 hist_data : list,
                 colors : list,
                 title : str,
                 name_graph: list = None, 
                 show_curve: bool=True,
                 show_hist: bool=True,
                 bin_size: int=1,
                 ) -> "KdeFactory":
      """
      Instantiates a KdeFactory instance with passed data and generates a kde plot

      Parameters
      ----------
      hist_data : list
          List of data arrays for which the KDE will be computed and plotted.
      colors : list
          List of colors for each dataset in hist_data.
      title : str
          Title of the plot.
      name_graph : list, optional
          List of names for each dataset in hist_data (default is None).
      show_curve : bool, optional
          If True, display the KDE curve on the plot (default is True).
      show_hist : bool, optional
          If True, display the histogram bars on the plot (default is True).
      bin_size : int, optional
          Size of the bins for the histogram (default is 1).

      Returns
      -------
      KdeFactory

      Notes 
      ------
      Additional plots can be plotted by passing dictionnaries of parameters in the format: TODO
      dict(
          type="scatter",
          x=curve_x[index],
          y=curve_y[index],
          xaxis="x1",
          yaxis="y1",
          mode="lines",
          #name=group_labels[index],
          #legendgroup=group_labels[index],
          showlegend=True,
          marker=dict(color=colors[index % len(colors)]),
          )
      """
      if not _is_array_like(name_graph) & (name_graph is not None):
        name_graph = [name_graph]
      
      if not _is_array_like(colors):
        colors = [colors]

      if not _is_array_like(hist_data[0]):
        hist_data = [hist_data]

      trace_number = len(hist_data)

      curve = [None] * len(hist_data)
      curve_x = [None] * trace_number
      curve_y = [None] * trace_number

      start = []
      end = []

      for trace in hist_data:
          start.append(min(trace) * 1.0)
          end.append(max(trace) * 1.0)

      absolute_min = min(start)
      absolute_max = max(end)

      for index in range(len(hist_data)):
          curve_x[index] = [absolute_min + x * (absolute_max - absolute_min) / 1000 for x in range(1000)] #TODO
          curve_y[index] = stats.gaussian_kde(hist_data[index])(curve_x[index])

      for ind, index in enumerate(range(len(hist_data))):
          curve[index] = dict(
              type="scatter",
              x=curve_x[index],
              y=curve_y[index],
              #xaxis="x1",
              #yaxis="y1",
              mode="lines",
              name=f"Graph - {ind}",
              legendgroup=ind,
              showlegend=True,
              marker=dict(color=colors[index % len(colors)]),
          )

          if name_graph is not None: 
            curve[index]["name"] = name_graph[index]
            
      ir = (absolute_max - absolute_min)
      viz = KdeFactory(
        hist_data=hist_data,
        graph_data=curve,
        start = start,
        end = end,
        colors = colors,
        bin_size = bin_size
      )

      return viz.plot(show_curve=show_curve, show_hist = show_hist, title=title)
