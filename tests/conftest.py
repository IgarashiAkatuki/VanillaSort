import numpy as np
import pytest
import torch
from spikeinterface.core import NumpyRecording

from vanillasort.checkpoints import load_config
from vanillasort.models import VanillaDet, HuiduRep


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory):
    """Random tiny weights exercise the real inference path without downloading."""
    config = load_config()
    config["detector_architecture"].update(
        d_model=8, nhead=2, num_layers=1, dim_ff=16, frontend_channels=8, max_len=5000
    )
    config["huidurep_architecture"].update(embedding_dim=16, n_heads=2, ff_dim=16, num_layers=3)
    config["chunk_len"] = 128
    config["profiles"]["d1"].update(base_threshold=0.0001, direct_keep=0.0001)
    # Exercise valid cross-fitted templates on a small fixture.
    config["residual"].update(min_core_events=2, max_template_events=32, fold_block_seconds=0.005, core_posterior=0.5)
    path = tmp_path_factory.mktemp("models") / "tiny.pt"
    with torch.random.fork_rng():
        torch.manual_seed(0)
        torch.save(
            {
                "config": config,
                "detector_state_dict": VanillaDet(**config["detector_architecture"]).state_dict(),
                "huidurep_state_dict": HuiduRep(**config["huidurep_architecture"]).state_dict(),
            },
            path,
        )
    return path


@pytest.fixture
def recording():
    rng = np.random.default_rng(10)
    traces = rng.normal(size=(1025, 4)).astype("float32")
    traces[100::150] -= np.array([8, 5, 3, 1], dtype="float32")
    recording = NumpyRecording([traces, traces[:768].copy()], 30000, channel_ids=["a", "b", "c", "d"])
    recording.set_dummy_probe_from_locations(np.array([[0, 0], [20, 0], [0, 20], [20, 20]]))
    return recording
