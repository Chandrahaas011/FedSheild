"""Custom Flower Strategy implementing communication-efficient FedAvg.

Serves both the FedAvg baseline (topk_ratio <= 0 -> plain dense FedAvg) and "your-method"
(topk_ratio in (0,1] -> clients send real sparse top-k updates, this class reconstructs
the dense delta server-side) from one implementation, so both configs share identical
aggregation logic and only differ in the client's fit() sparsification step
(src/fl/client.py). Bytes/round is measured from the actual ndarrays clients returned,
not simulated.
"""
import numpy as np
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from src.fl.param_utils import dense_bytes, flatten, unflatten


class CommEfficientFedAvg(FedAvg):
    def __init__(self, initial_ndarrays: list[np.ndarray], topk_ratio: float = -1, **kwargs):
        self.model_shapes = [a.shape for a in initial_ndarrays]
        flat, sizes = flatten(initial_ndarrays)
        self.sizes = sizes
        self.global_flat = flat
        self.topk_ratio = topk_ratio
        self.round_bytes: dict[int, int] = {}
        self.cumulative_bytes = 0
        kwargs["initial_parameters"] = ndarrays_to_parameters(initial_ndarrays)
        super().__init__(**kwargs)

    def aggregate_fit(self, server_round: int, results: list[tuple[ClientProxy, FitRes]], failures):
        if not results:
            return None, {}
        total_examples = sum(fit_res.num_examples for _, fit_res in results)
        total_bytes = 0

        if self.topk_ratio and self.topk_ratio > 0:
            agg_delta = np.zeros_like(self.global_flat)
            for _, fit_res in results:
                ndarrays = parameters_to_ndarrays(fit_res.parameters)
                idx, values = ndarrays[0].astype(np.int64), ndarrays[1].astype(np.float32)
                weight = fit_res.num_examples / total_examples
                np.add.at(agg_delta, idx, values * weight)
                total_bytes += int(fit_res.metrics.get("bytes_sent", idx.nbytes + values.nbytes))
            new_flat = self.global_flat + agg_delta
        else:
            weighted_sum = np.zeros_like(self.global_flat)
            for _, fit_res in results:
                ndarrays = parameters_to_ndarrays(fit_res.parameters)
                flat, _ = flatten(ndarrays)
                weight = fit_res.num_examples / total_examples
                weighted_sum += flat * weight
                total_bytes += int(fit_res.metrics.get("bytes_sent", dense_bytes(ndarrays)))
            new_flat = weighted_sum

        self.global_flat = new_flat
        self.round_bytes[server_round] = total_bytes
        self.cumulative_bytes += total_bytes

        new_ndarrays = unflatten(new_flat, self.model_shapes, self.sizes)
        metrics_agg: dict[str, Scalar] = {
            "bytes_this_round": total_bytes,
            "cumulative_bytes": self.cumulative_bytes,
        }
        eps_vals = [fit_res.metrics["dp_epsilon"] for _, fit_res in results if "dp_epsilon" in fit_res.metrics]
        if eps_vals:
            metrics_agg["dp_epsilon_max"] = max(eps_vals)
        loss_vals = [(fit_res.num_examples, fit_res.metrics.get("train_loss", 0.0)) for _, fit_res in results]
        if loss_vals:
            metrics_agg["train_loss"] = sum(n * l for n, l in loss_vals) / total_examples

        return ndarrays_to_parameters(new_ndarrays), metrics_agg

    def aggregate_evaluate(self, server_round: int, results, failures):
        if not results:
            return None, {}
        total_examples = sum(eval_res.num_examples for _, eval_res in results if eval_res.num_examples > 0)
        if total_examples == 0:
            return None, {}
        loss = sum(eval_res.num_examples * eval_res.loss for _, eval_res in results) / total_examples
        macro_f1 = sum(
            eval_res.num_examples * eval_res.metrics.get("macro_f1", 0.0) for _, eval_res in results
        ) / total_examples
        accuracy = sum(
            eval_res.num_examples * eval_res.metrics.get("accuracy", 0.0) for _, eval_res in results
        ) / total_examples
        return loss, {"macro_f1": macro_f1, "accuracy": accuracy}
