from contextlib import contextmanager

import mlflow

MLRUNS_DIR = "mlruns"


@contextmanager
def mlflow_run(experiment_name: str, run_name: str, config: dict):
    mlflow.set_tracking_uri(f"file:{MLRUNS_DIR}")
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        flat = _flatten(config)
        for k, v in flat.items():
            mlflow.log_param(k, v)
        yield run


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, prefix=f"{key}."))
        else:
            out[key] = v
    return out


def log_round_metrics(metrics: dict, step: int) -> None:
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            mlflow.log_metric(k, v, step=step)
