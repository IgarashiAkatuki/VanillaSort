"""Versioned, local checkpoint resolution. Importing this module never downloads weights."""

from __future__ import annotations

import hashlib
import json
import os
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


def resolve_model(model="default", model_path=None):
    """Resolve bundled weights, a local directory, or a combined training checkpoint.

    A combined .pt contains ``detector_state_dict``, ``huidurep_state_dict``
    and optionally ``config`` (architecture/settings overrides). A directory
    contains config.json plus the two checkpoint paths given in that config.
    No files are downloaded. An optional VANILLASORT_MODEL_CACHE directory can
    hold an exact copy of the bundled files under MODEL_VERSION/.
    """
    if model != "default":
        raise ValueError(f"Unknown model {model!r}; use 'default' and optionally model_path")
    config = load_config()
    if model_path is None:
        cache = os.environ.get("VANILLASORT_MODEL_CACHE")
        if cache:
            directory = Path(cache).expanduser() / MODEL_VERSION
            for key in CHECKPOINT_HASHES:
                cached = directory / Path(config[key]).name
                if cached.is_file():
                    config[key] = str(cached)
        for key, expected in CHECKPOINT_HASHES.items():
            if file_hash(config[key]) != expected:
                raise ValueError(f"Default checkpoint checksum mismatch: {config[key]}")
        metadata = {"version": MODEL_VERSION, "source": MODEL_SOURCE}
    else:
        path = Path(model_path).expanduser().resolve()
        if path.is_dir():
            settings = path / "config.json"
            if not settings.is_file():
                raise ValueError("A model directory must contain config.json and both model checkpoints")
            local = json.loads(settings.read_text(encoding="utf-8"))
            if not all(key in local for key in CHECKPOINT_HASHES):
                raise ValueError("config.json must specify detector_checkpoint and huidurep_checkpoint")
            config.update(local)
            for key in CHECKPOINT_HASHES:
                config[key] = str((path / local[key]).resolve())
        elif path.is_file():
            import torch

            bundle = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(bundle, dict) or not all(
                key in bundle for key in ("detector_state_dict", "huidurep_state_dict")
            ):
                raise ValueError("A .pt model bundle must contain detector_state_dict and huidurep_state_dict")
            config.update(bundle.get("config", {}))
            for key in CHECKPOINT_HASHES:
                config[key] = str(path)
        else:
            raise FileNotFoundError(f"Model path does not exist: {path}")
        metadata = {"version": "local", "source": str(path)}
    for key in CHECKPOINT_HASHES:
        if not Path(config[key]).is_file():
            raise FileNotFoundError(config[key])
    if config["detector_architecture"].get("num_channels", 4) != 4:
        raise ValueError("This pipeline requires a four-channel VanillaDet checkpoint")
    metadata["sha256"] = {key: file_hash(config[key]) for key in CHECKPOINT_HASHES}
    return config, metadata


def verify_bundle():
    config = load_config()
    for key, expected in CHECKPOINT_HASHES.items():
        path = Path(config[key])
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"Bundled checkpoint changed or missing: {path}")
    return {"verified_files": len(CHECKPOINT_HASHES), "algorithm": "sha256", "model_version": MODEL_VERSION}


def _state(path, component):
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    if component in state:
        state = state[component]
    if "state_dict" in state:
        state = state["state_dict"]
    elif "model_state_dict" in state:
        state = state["model_state_dict"]
    return {k.removeprefix("module."): v for k, v in state.items()}


def load_detector(config, device):
    from .models import VanillaDet

    model = VanillaDet(**config["detector_architecture"])
    model.load_state_dict(_state(config["detector_checkpoint"], "detector_state_dict"), strict=True)
    return model.to(device).eval()


def load_huidurep(config, device):
    from .models import HuiduRep

    model = HuiduRep(**config["huidurep_architecture"])
    model.load_state_dict(_state(config["huidurep_checkpoint"], "huidurep_state_dict"), strict=True)
    return model.to(device).eval()
