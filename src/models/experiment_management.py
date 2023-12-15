import mlflow.data
from mlflow_extend import mlflow
from mlflow.models import Model, infer_signature, ModelSignature
from mlflow.pyfunc import PyFuncModel
from mlflow import pyfunc
from mlflow.tracking import MlflowClient
import mlflow.sklearn

import databricks
from databricks.feature_store import FeatureStoreClient
from databricks.feature_store import FeatureLookup

def log_mlflow(experiment_name : str, run_name : str, func_params : dict, score : float, pred_model = None) -> None:

    try: 
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        runs = mlflow.search_runs(experiment_ids=[experiment_id])
        iteration = len(runs)
    except:
        iteration = 0

    loggged_name = f"{experiment_name}_{run_name}_V{iteration}"

    # Start an MLflow run
    with mlflow.start_run(run_name=loggged_name):

        mlflow.log_params(func_params)
        mlflow.log_metric("score", score)

        if pred_model is not None:
            mlflow.sklearn.log_model(model, "random_forest_model")
        
    mlflow.end_run()
