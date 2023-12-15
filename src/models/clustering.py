import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.model_selection import TimeSeriesSplit

def cluster_gmm(data : np.ndarray, n_components : int = 4, n_clusters : int = 4) -> tuple:

    gmm = GaussianMixture(n_components = n_components, random_state = 0).fit(data)

    return (gmm.predict_proba(data), gmm.bic(data))

def cluster_kmeans(data : np.ndarray, n_clusters : int = 4, n_components : int = 4) -> tuple:
    
    kmeans = KMeans(n_clusters = n_clusters).fit(data)

    return (kmeans.predict(data), silhouette_score(data, kmeans.labels_))

class PredictiveClustering:

    def __init__(self, data : np.ndarray, clustering_model, predicting_model):
        self.data = data
        self.clustering_model = clustering_model
        self.predicting_model = predicting_model

        self.full_predictions = None
        
    def feed_forward_clustering(self, clustering_params : dict, split_index : int, retrain_step : int = 10) -> pd.DataFrame:

        # train/test split and initial model training
        init_train_array = self.data[:split_index]
        test_array = self.data[split_index:]

        rd_model = self.clustering_model(**clustering_params).fit(init_train_array)
        in_sample_pred = rd_model.predict(init_train_array)
        
        # predict the state of the next observation
        states_pred = []
        for i in range(len(test_array)):
            
            preds = rd_model.predict(self.data[split_index])
            states_pred.append(preds)
            
            # retrain the existing model
            if i % retrain_step == 0:
                rd_model = self.clustering_model(**clustering_params).fit(self.data[:split_index])
            
            split_index += 1

        in_sample = pd.DataFrame(data = in_sample_pred, columns = ["predictions"]).assign(ticker = "in_sample")
        out_sample = pd.DataFrame(data = states_pred, columns = ["predictions"]).assign(ticker = "out_sample")

        full_predictions = pd.concat([in_sample, out_sample], axis = 0)

        self.full_predictions =  full_predictions

        return self
    
    def feed_forward_predicting(self, kwargs_pred : dict, kwargs_cluster : dict, initial_split_frac : float = 0.75, retrain_step : int = 10) -> pd.DataFrame:

        # train/test split and initial model training
        split_index = int(len(self.data) * initial_split_frac)
        nb_steps = len(self.data[split_index:])

        # predict the state of the next observation

        y = self.feed_forward_clustering(**kwargs_cluster)["predictions"].to_numpy()

        res = []
        for i in range(0, nb_steps, retrain_step):

            split_index_lower = split_index + i
            split_index_upper = split_index + i + retrain_step

            X_train, y_train = self.data[:split_index_lower], y[:split_index_lower]
            X_test, y_test = self.data[split_index_lower : split_index_upper], y[split_index_lower : split_index_upper]

            pred_model = self.predicting_model(**kwargs_pred).fit(X_train, y_train)
            res_accuracy = pred_model.score(X_test, y_test)
            res.append(res_accuracy)

        return res

    def make_cross_val(self, 
                       cluster_params : dict, 
                       predict_params : dict,
                       n_splits : int = 5):

        tscv = TimeSeriesSplit(n_splits=n_splits)

        X = self.data.to_numpy()
        y = self.clustering_model(**cluster_params).fit_predict(X)

        pred_model = self.predicting_model(**predict_params)

        cv_results = []
        for i, (train_index, test_index) in enumerate(tscv.split(X)):

            y_train, y_test = y[train_index], y[test_index]
            X_train, X_test = X[train_index], X[test_index]

            pred_model.fit(X_train, y_train)
            
            y_pred_probs = pred_model.predict_proba(X_test)
            y_pred = np.argmax(y_pred_probs, axis = 1)
            accuracy = accuracy_score(y_test, y_pred)

            cv_results.append(accuracy)
        
        return np.mean(cv_results)
    







