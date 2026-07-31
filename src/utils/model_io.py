"""Loads a trained model back from disk for a given run_id + its config — shared by the
experiment runner (final full-metric evaluation) and the XAI entrypoint (explaining the
converged model)."""
from pathlib import Path

import numpy as np
import torch

from src.fl.dp import make_dp_compatible
from src.models.build import build_model
from src.utils.checkpoint import load_checkpoint


def load_trained_model(run_id: str, config: dict, n_features: int, n_classes: int) -> torch.nn.Module:
    model = build_model(config["model"], n_features, n_classes)
    if config["method"] == "centralized":
        ckpt = load_checkpoint(Path("checkpoints") / run_id / "centralized.pt")
        model.load_state_dict(ckpt["model_state"])
    else:
        if config.get("dp", False) or config["method"] == "topk_dp":
            model = make_dp_compatible(model)
        params_path = Path("checkpoints") / run_id / "global_params.npz"
        loaded = np.load(params_path)
        state_dict = model.state_dict()
        for k, arr_name in zip(state_dict.keys(), loaded.files):
            state_dict[k] = torch.tensor(loaded[arr_name])
        model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model
