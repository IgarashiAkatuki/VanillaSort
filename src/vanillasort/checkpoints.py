"""Versioned, local checkpoint resolution. Importing this module never downloads weights."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent
MODEL_VERSION = "hybrid-janelia-2026.09"
MODEL_SOURCE = "https://github.com/IgarashiAkatuki/VanillaSort/tree/00a9ef06e88a955406ad2902d7bc56617f0afd2e"
CHECKPOINT_HASHES = {
    "detector_checkpoint": "c112499b0077d3613ef5b791f61130eacb75d090bea4f6fa9d59aabc0c2aa1a6",
    "huidurep_checkpoint": "048203dbb326e159f4320bc4a7204c93a9951477b8c9995a75728bb87378da68",
}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config():
    config = json.loads((PACKAGE / "configs/default.json").read_text(encoding="utf-8"))
    for key in CHECKPOINT_HASHES:
        config[key] = str(PACKAGE / config[key])
    return config


def verify_bundle():
    config = load_config()
    for key, expected in CHECKPOINT_HASHES.items():
        path = Path(config[key])
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"Bundled checkpoint changed or missing: {path}")
    return {"verified_files": len(CHECKPOINT_HASHES), "algorithm": "sha256", "model_version": MODEL_VERSION}


def _state(path):
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    elif "model_state_dict" in state:
        state = state["model_state_dict"]
    return {k.removeprefix("module."): v for k, v in state.items()}


def load_detector(config, device):
    from .models import VanillaDet

    model = VanillaDet(**config["detector_architecture"])
    model.load_state_dict(_state(config["detector_checkpoint"]), strict=True)
    return model.to(device).eval()


def load_huidurep(config, device):
    from .models import HuiduRep

    model = HuiduRep(**config["huidurep_architecture"])
    model.load_state_dict(_state(config["huidurep_checkpoint"]), strict=True)
    return model.to(device).eval()
