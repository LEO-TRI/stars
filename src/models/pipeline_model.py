import pandas as pd
import numpy as np
import random
import plotly.express as px 

class ModelTest:

    def __init__(self, data : np.ndarray) -> None:

        self.data = data

        if isinstance(data, (pd.DataFrame, pd.Series)):
            self.index = data.index.to_list()
        else: 
            self.index = list(range(len(data.index)))
        
        self.processed_data = None
        self.predictions = None
        self.params_func = None
        self.score = None

        self.log_all = {"params" : [], "predictions" : [], "score" : []}
        self.log_sanity_checks = []
    
    def add_model(self, params_func : dict):
        
        if not isinstance(params_func, dict):
           params_func = dict(returns=params_func[0],
                              smooth=params_func[1],
                              standardise=params_func[2],
                              clustering=params_func[3])
           
        self.params_func = params_func
        self.return_func = params_func.get("returns")[0]
        self.smoothing_func = params_func.get("smooth")[0]
        self.standardise_func = params_func.get("standardise")[0]
        self.clustering_func = params_func.get("clustering")[0]

        return self

    def make_preprocess(self):
        # Apply preprocessing functions sequentially

        processed_data = self.return_func(self.data)
        processed_data = self.smoothing_func(processed_data, **self.params_func.get("smooth")[1])
        processed_data = self.standardise_func(processed_data, **self.params_func.get("standardise")[1])

        self.processed_data = processed_data

        return self
    
    def make_fit_predict(self):

        self.predictions, self.score = self.clustering_func(self.processed_data, **self.params_func.get("clustering")[1])

        return self
    
    def make_sanity_check(self):

        predictions = (self.predictions).astype(str)
        if self.predictions.ndim > 1:
            predictions = np.argmax(self.predictions, axis=1).astype(str)
        
        processed_data = pd.DataFrame(self.processed_data).assign(predictions = predictions)
        
        res = []
        groups = []
        for val in processed_data.predictions.unique():
            res.append((processed_data.loc[lambda x : x.predictions == val]
                                      .select_dtypes(np.number)
                                      .mean(axis = 1)
                                      .describe().T))
            
            groups.append(val)

        #Log the result with a ticker
        res = (pd.DataFrame(res).assign(**self.params_func.get("smooth")[1])
                                .assign(**self.params_func.get("standardise")[1])
                                .assign(**self.params_func.get("clustering")[1])
                                .assign(groups = groups)
                                .set_index("groups"))
                
        self.log_sanity_checks.append(res)
    
        return self

    def plot(self, data : np.ndarray, predictions : np.ndarray, palette: list = None):

        predictions_str = predictions.astype(str)
        if predictions.ndim > 1:
            predictions_str = (np.argmax(predictions, axis=1)).astype(str)

        n_unique_clusters = len(np.unique(predictions_str))

        if palette is None:
            palette = px.colors.qualitative.G10

        #Slice the part of the data lost in the rolling preprocessing to have equal length
        data = data.iloc[len(data) -  len(predictions):]

        return px.scatter(x = data.index, 
                          y = np.mean(data.select_dtypes(np.number).to_numpy(), axis = 1), 
                          title = f"Market clustering with {n_unique_clusters} clusters",
                          color = predictions_str, 
                          color_discrete_sequence = palette, 
                          template="plotly_white")
    
    def make_run_all(self, params_func: dict, palette : list = None, show_fig : bool = False):
        
        self.add_model(params_func).make_preprocess().make_fit_predict()
        self.make_sanity_check()

        self.log_all.get("params").append(params_func)
        self.log_all.get("predictions").append(self.predictions)
        self.log_all.get("score").append(self.score)

        if show_fig:
            fig = self.plot(self.data, self.predictions, palette)
            fig.show()

        return self 
    
class RandomTuner(ModelTest):

    def __init__(self, data : np.ndarray, cluster_params : list[tuple]):

        self.data = data
        self.cluster_params = cluster_params

        self.processed_data = None
        self.predictions = None
        self.params_func = None
        self.score = None

        self.log_all = {"params" : [], "predictions" : [], "score" : []}
        self.log_sanity_checks = []

    def random_tuning(self, frac_iterations: float = 0.3):

        cluster_params = self._permut_to_dict(cluster_params)

        n_combinations = len(cluster_params)

        iterations = int(n_combinations * frac_iterations)

        shuffled_params = random.sample(cluster_params, n_combinations)

        for elem in shuffled_params[:iterations]:

            self.make_run_all(elem)
        
        return self 
    
    def _permut_to_dict(self, combinations : list[tuple]) -> list[dict]:
        """
        Converts a list of tuples to a list of dicts ready to be used in make_run_all()

        Parameters
        ----------
        combinations : list[tuple]
            List of tuples of parameters

        Returns
        -------
        list[dict]
            List of dictionaries that can be unloaded in make_run_all()
        """
        
        params_list =[]

        for combi in combinations:
            dict_combi = dict(returns=(combi[0], {}),
                              smooth=(combi[1], combi[2]),
                              standardise=(combi[3], combi[4]),
                              clustering=(combi[5], combi[6])
                            )
            
            params_list.append(dict_combi)
        
        return params_list




