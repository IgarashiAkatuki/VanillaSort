"""Pinned Hugging Face models and local checkpoints, with lazy download and SHA-256 checks."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
MODEL = json.loads((PACKAGE / "configs/default_model.json").read_text(encoding="utf-8"))
MODEL_VERSION = MODEL["version"]
CHECKPOINT_HASHES = {key: item["sha256"] for key, item in MODEL["files"].items()}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config():
    """Read inference settings without loading or downloading any model."""
    return json.loads((PACKAGE / "configs/default.json").read_text(encoding="utf-8"))


def _default_checkpoint(filename):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    revision = MODEL["revision"]
    if not isinstance(revision, str) or len(revision) != 40:
        raise RuntimeError("The default Hugging Face model release is not pinned; use model_path for local weights")
    cache = os.environ.get("VANILLASORT_MODEL_CACHE")
    options = dict(
        repo_id=MODEL["repo_id"],
        filename=filename,
        revision=revision,
        cache_dir=str(Path(cache).expanduser()) if cache else None,
        token=False,
        library_name="vanillasort",
    )
    try:
        # An immutable cached snapshot needs no metadata request or network.
        return hf_hub_download(**options, local_files_only=True)
    except LocalEntryNotFoundError:
        try:
            return hf_hub_download(**options)
        except Exception as exc:
            raise RuntimeError(
                f"Cannot download {filename} from {MODEL['repo_id']} at {revision}. "
                "Connect once to populate the Hugging Face cache, or supply model_path for offline use."
            ) from exc


def resolve_model(model="default", model_path=None):
    """Resolve Hugging Face weights, a local directory, or a combined training checkpoint.

    A combined .pt contains ``detector_state_dict``, ``huidurep_state_dict``
    and optionally ``config`` (architecture/settings overrides). A directory
    contains config.json plus the two checkpoint paths given in that config.
    Default weights download only on cache misses, from a pinned Hub commit.
    VANILLASORT_MODEL_CACHE overrides the standard Hugging Face cache directory.
    Local checkpoints never use the network.
    """
    if model != "default":
        raise ValueError(f"Unknown model {model!r}; use 'default' and optionally model_path")
    config = load_config()
    if model_path is None:
        for key, expected in CHECKPOINT_HASHES.items():
            config[key] = _default_checkpoint(MODEL["files"][key]["filename"])
            if file_hash(config[key]) != expected:
                raise ValueError(f"Default checkpoint checksum mismatch: {config[key]}")
        metadata = {
            "version": MODEL_VERSION,
            "source": f"https://huggingface.co/{MODEL['repo_id']}",
            "repo_id": MODEL["repo_id"],
            "revision": MODEL["revision"],
        }
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
    resolve_model()
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
