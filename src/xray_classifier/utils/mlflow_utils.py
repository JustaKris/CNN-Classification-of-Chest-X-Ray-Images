"""MLflow experiment tracking utilities."""

import mlflow
import mlflow.tensorflow
from mlflow import log_metric, log_param


def setup_mlflow(experiment_name):
    """Set the active experiment and start a new MLflow run."""
    mlflow.set_experiment(experiment_name)
    mlflow.start_run()


def log_mlflow_params(params):
    """Log a dictionary of parameters to the active MLflow run."""
    for key, value in params.items():
        log_param(key, value)


def log_mlflow_metrics(metrics):
    """Log a dictionary of metrics to the active MLflow run."""
    for key, value in metrics.items():
        log_metric(key, value)


def end_mlflow_run():
    """End the current active MLflow run."""
    mlflow.end_run()
