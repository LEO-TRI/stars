import pandas as pd
import numpy as np
import copy

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import plotly.express as px 
import plotly.graph_objects as go

class FeatureExplainer():

    def __init__(self, model, features : list, coefs = None) -> tuple:
        self.model = model
        self.features = features
        self.coefs = coefs


        self.permut_coefs = None
        self.fig = None

    @staticmethod
    def check_has_importance(model) -> tuple:
        """
        _summary_

        Parameters
        ----------
        model : _type_
            _description_

        Returns
        -------
        tuple
            _description_
        """
        
        try:
            features = model[:-1].get_feature_names_out()
        except:
            features = None

        if hasattr(model, "coef_"):
        
            coefs = model.coef_
            #coef_df = pd.DataFrame(np.vstack((features, coefs)).T, columns = ["features", "coefs"])
            #coef_df = coef_df.astype({"coefs": float, "features": str}).sort_values("coefs", ascending=False)
            #features = coef_df["features"].to_numpy()
            #coefs = coef_df["coefs"].to_numpy()

        else: 
            coefs = None
        
        return (features, coefs)

    @classmethod
    def from_model(cls, 
                   model, 
                   is_show: bool = False) -> "FeatureExplainer":

        features, coefs = cls.check_has_importance(model)
        fe = FeatureExplainer(model, features, coefs)

        if is_show:
            fig = fe.plot_top_features(fe.coefs)
            fig.show()

        return fe 
        
    @classmethod
    def from_train(cls, 
                   data : tuple,
                   model, 
                   model_params : dict = None, 
                   test_split: float = 0.3, 
                   is_show: bool = False) -> "FeatureExplainer":
        """
        _summary_

        Parameters
        ----------
        data : tuple
            _description_
        model : _type_
            _description_
        model_params : dict, optional
            _description_, by default None
        test_split : float, optional
            _description_, by default 0.3
        is_show : bool, optional
            _description_, by default False

        Returns
        -------
        FeatureExplainer
            _description_
        """
        
        X, y = data
        X, y = _series_to_array(X), _series_to_array(y)

        assert len(X) == len(y)

        split_index = int(len(X) * test_split)
        X_train, y_train = X[:split_index], y[:split_index]

        if model_params is None: 
            model_trained = model()
        else:
            model_trained = model(**model_params)
        model_trained.fit(X_train, y_train)

        fe = cls.from_model(model_trained)

        if is_show:
            fig = fe.plot_top_features(fe.coefs)
            fig.show()

        return fe 
    
    def compute_permutation(self, 
                            data: tuple, 
                            n_repeat: int = 10, 
                            is_show: bool = False,
                            test_split: float = 0.3,
                            #threshold: float = 0.5,
                            n_cols : int = None,
                            score_func = accuracy_score) -> tuple[np.ndarray]:
        """
        Compute the feature importance for each col

        Parameters
        ----------
        data : tuple
            _description_
        n_repeat : int, optional
            _description_, by default 5
        is_show : bool, optional
            _description_, by default False
        threshold : float, optional
            _description_, by default 0.5
        n_cols : int, optional
            _description_, by default None
        score_func : _type_, optional
            _description_, by default accuracy_score

        Returns
        -------
        tuple[np.ndarray]
            _description_
        """
        
        X, y = data
        features = X.columns.to_list() 

        if isinstance(y, (np.ndarray, list)):
            y = pd.Series(y)
        if n_cols is not None: 
            features = features[:n_cols]

        split_index = int(len(X) * test_split)
        X_test, y_test = X.iloc[split_index:], y.iloc[split_index:]

        #y_pred_proba = self.model.predict_proba(X_test)[:,1]
        #y_pred = np.where(y_pred_proba>=threshold, 1, 0)

        X_test = _series_to_array(X_test)

        y_pred_proba = self.model.predict_proba(X_test)
        y_pred = np.argmax(y_pred_proba, axis = 1)

        score = score_func(y_test, y_pred)

        results_lines = []

        for i, feature in enumerate(features):
            
            shuffled_df = copy.deepcopy(X_test)

            #TODO Add a feature transfo step

            print(f"Testing column {feature}")

            repeat_range = [] 

            for _ in range(n_repeat):

                np.random.default_rng().shuffle(shuffled_df[:,i])

                y_pred_proba = self.model.predict_proba(shuffled_df)
                y_pred = np.argmax(y_pred_proba, axis = 1)

                #y_pred_proba = self.model.predict_proba(X_test)[:,1]
                #y_pred = np.where(y_pred_proba>=threshold, 1, 0)

                repeat_range.append(score_func(y_test, y_pred))

            results_lines.append(repeat_range)
        
        results_lines = np.array(results_lines)
        res_features_cv = np.mean(results_lines, axis=1)

        res_final = score - res_features_cv 

        if is_show:
            self.plot_top_columns(features, res_final)
            self.fig.show()

        self.permut_coefs = res_final
        self.cols = features
        self.results = pd.DataFrame({"features" : features, "diff_score" : res_final})

        return self


    def plot_top_features(self, coefs):

        self.fig = feature_importance_plotting(coefs)

        return self.fig
    
    def plot_top_columns(self, cols, coefs):

        self.fig = feature_cols_plotting(cols, coefs)

        return self


def feature_importance_plotting(features: np.ndarray) -> go.Scatter:
    """
    Plot the feature importance for each feature

    Parameters
    ----------
    features : np.ndarray
        _description_

    Returns
    -------
    go.Scatter
        _description_
    """

    index = np.ones(len(features))
    fig = px.scatter(y=index, x=features)
    fig.update_layout(title="Feature importance Plot",
                      xaxis_title="Feature Importance",
                      yaxis_title="")
    return fig

def feature_cols_plotting(features: np.ndarray, coefs_permut : np.ndarray) -> go.Scatter:
    """
    Plot the column importance for each feature 

    Parameters
    ----------
    features : np.ndarray
        _description_

    Returns
    -------
    go.Scatter
        _description_
    """

    fig = px.bar(y=coefs_permut, x=features)
    fig.update_layout(title="Feature importance Plot",
                      xaxis_title="Columns",
                      yaxis_title="Feature Importance")
    return fig

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
    if isinstance(obj, (pd.Series, pd.DataFrame)): 
      return obj.to_numpy()
    elif isinstance(obj, list):
      return np.array(obj)
    return obj