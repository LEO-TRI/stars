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
        
        features = model[:-2].get_feature_names_out()

        if hasattr(model[-1], "feature_importances_"):
        
            coefs = model[-1].feature_importances_
            coef_df = pd.DataFrame(np.vstack((features, coefs)).T, columns = ["features", "coefs"])
            coef_df = coef_df.astype({"coefs": float, "features": str}).sort_values("coefs", ascending=False)
            features = coef_df["features"].to_numpy()
            coefs = coef_df["coefs"].to_numpy()

        else: 
            coefs = None
        
        return (features, coefs)

    @classmethod
    def from_model(cls, 
                   model, 
                   is_show: bool = False) -> "FeatureExplainer":

        features, coefs = cls.check_has_importance(model)
        fe = FeatureExplainer(model, features, coefs)

        fig = fe.plot_top_features(fe.coefs)
        if is_show:
            fig.show()

        return fe 
        
    @classmethod
    def from_train(cls, 
                   data : tuple,
                   model, 
                   model_params : dict = None, 
                   test_split: float = 0.3, 
                   is_show: bool = False) -> "FeatureExplainer":
        """_summary_

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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=1830, stratify=y)

        if model_params is None: 
            model_trained = model().fit(X_train, y_train)
        else:
            model_trained = model(**model_params).fit(X_train, y_train)

        fe = cls.from_model(model_trained)

        fig = fe.plot_top_features(fe.coefs)
        if is_show:
            fig.show()

        return fe 
    
    def compute_permutation(self, 
                            data: tuple, 
                            n_repeat: int = 5, 
                            is_show: bool = False,
                            threshold: float = 0.5,
                            n_cols : int = None,
                            score_func = accuracy_score) -> tuple[np.ndarray]:
        """_summary_

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
        
        X_test, y_test = data
        
        features = X_test.columns.to_list() 
        if n_cols is not None: 
            features = features[:n_cols]

        #y_pred_proba = self.model.predict_proba(X_test)[:,1]
        #y_pred = np.where(y_pred_proba>=threshold, 1, 0)

        y_pred_proba = self.model.predict_proba(shuffled_df)
        y_pred = np.argmax(y_pred_proba, axis = 1)

        score = score_func(y_test, y_pred)

        results_lines = []

        for i, feature in enumerate(features):
            
            shuffled_df = copy.deepcopy(X_test)

            #TODO Add a feature transfo step

            print(f"Testing column {feature}")

            repeat_range = [] 

            for _ in range(n_repeat):

                shuffled_df[feature] = shuffled_df[feature].sample(frac=1, replace = False).to_numpy()
                y_pred_proba = self.model.predict_proba(shuffled_df)
                y_pred = np.argmax(y_pred_proba, axis = 1)

                #y_pred_proba = self.model.predict_proba(X_test)[:,1]
                #y_pred = np.where(y_pred_proba>=threshold, 1, 0)

                repeat_range.append(score_func(y_test, y_pred))

            results_lines.append(repeat_range)
        
        results_lines = np.array(results_lines)
        res_features_cv = np.mean(results_lines, axis=1)

        res_final = score - res_features_cv 

        self.fig = self.plot_top_columns(features, res_final)
        if is_show:
            self.fig.show()

        self.permut_coefs = res_final
        self.cols = features

        return (features, res_final)


    def plot_top_features(self, coefs):

        self.fig = feature_importance_plotting(coefs)

        return self.fig
    
    def plot_top_columns(self, cols, coefs):

        self.fig = feature_cols_plotting(cols, coefs)
        return self.fig


def feature_importance_plotting(features: np.ndarray) -> go.Scatter:
    """
    _summary_

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
    _summary_

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
