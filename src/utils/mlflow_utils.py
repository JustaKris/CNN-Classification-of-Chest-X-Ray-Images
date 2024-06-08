import mlflow
import mlflow.tensorflow
from mlflow import log_metric, log_param, log_artifacts

def setup_mlflow(experiment_name):
    mlflow.set_experiment(experiment_name)
    mlflow.start_run()

def log_mlflow_params(params):
    for key, value in params.items():
        log_param(key, value)

def log_mlflow_metrics(metrics):
    for key, value in metrics.items():
        log_metric(key, value)

def end_mlflow_run():
    mlflow.end_run()
