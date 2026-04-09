import os
from typing import Any, Dict, Tuple

from smax_ctm.analysis.io_utils import load_pickle
from smax_ctm.analysis.types import AnalysisCheckpoint


def load_ctm_checkpoint(checkpoint_path: str) -> AnalysisCheckpoint:
    if not checkpoint_path:
        raise ValueError("checkpoint_path must be non-empty")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    raw = load_pickle(checkpoint_path)
    if "config" not in raw:
        raise KeyError("Checkpoint missing required key: config")
    if "actor_params" not in raw:
        raise KeyError("Checkpoint missing required key: actor_params")

    model_type = raw.get("model_type", None)
    if model_type is not None and model_type != "ctm":
        raise ValueError(f"Expected CTM checkpoint, got model_type={model_type}")

    return AnalysisCheckpoint(
        path=checkpoint_path,
        config=raw["config"],
        actor_params=raw["actor_params"],
    )


def extract_ctm_and_head_params(actor_params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if "params" not in actor_params:
        raise KeyError("actor_params missing top-level key: params")

    root = actor_params["params"]
    if "ScannedCTM_0" not in root:
        raise KeyError(
            "actor_params['params'] missing ScannedCTM_0. "
            f"Available keys: {sorted(root.keys())}"
        )

    scanned = root["ScannedCTM_0"]
    if "CTMCell_0" not in scanned:
        raise KeyError(
            "actor_params['params']['ScannedCTM_0'] missing CTMCell_0. "
            f"Available keys: {sorted(scanned.keys())}"
        )

    ctm_params = {"params": scanned["CTMCell_0"]}

    required_dense = ["Dense_0", "Dense_1", "Dense_2"]
    missing_dense = [k for k in required_dense if k not in root]
    if missing_dense:
        raise KeyError(
            "Actor head params missing required Dense layers: "
            f"{missing_dense}. Available keys: {sorted(root.keys())}"
        )

    head_params = {
        "Dense_0": root["Dense_0"],
        "Dense_1": root["Dense_1"],
        "Dense_2": root["Dense_2"],
    }
    return ctm_params, head_params
