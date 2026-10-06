import json
import subprocess
import sys

import numpy as np
import pytest
import torch
from spikeinterface.core import BaseSorting, NumpyRecording, load

import vanillasort
from vanillasort.checkpoints import resolve_model
from vanillasort.geometry import local_neighborhoods, deduplicate_events


def test_import():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, vanillasort; assert vanillasort.__version__; assert callable(vanillasort.sort); assert 'torch' not in sys.modules; assert 'huggingface_hub' not in sys.modules",
        ],
        check=True,
    )


def test_sort_roundtrip(recording, tiny_model, tmp_path):
    rng_before = torch.get_rng_state().clone()
    threads = torch.get_num_threads()
    folder = tmp_path / "result"
    sorting = vanillasort.sort(
        recording, model_path=tiny_model, components=2, device="cpu", verbose=False, output_folder=folder
    )
    assert isinstance(sorting, BaseSorting)
    assert sorting.get_num_segments() == 2
    assert sorting.get_sampling_frequency() == 30000
    spikes = sorting.to_spike_vector()
    assert spikes.size > 0
    assert spikes["sample_index"].dtype.kind in "iu"
    assert set(spikes["segment_index"]) == {0, 1}
    for segment in range(2):
        selected = spikes[spikes["segment_index"] == segment]
        assert np.all(np.diff(selected["sample_index"]) >= 0)
        assert np.all(selected["sample_index"] >= 30)
        assert np.all(selected["sample_index"] < recording.get_num_samples(segment) - 30)
    assert np.isin(sorting.get_property("main_channel_id"), recording.channel_ids).all()
    assert not set(spikes[spikes["segment_index"] == 0]["unit_index"]) & set(
        spikes[spikes["segment_index"] == 1]["unit_index"]
    )
    restored = load(folder / "sorting")
    np.testing.assert_array_equal(restored.to_spike_vector(), spikes)
    np.testing.assert_array_equal(restored.get_property("main_channel_id"), sorting.get_property("main_channel_id"))
    with np.load(folder / "events.npz") as events:
        np.testing.assert_array_equal(events["sample_index"], spikes["sample_index"])
        np.testing.assert_array_equal(events["unit_id"], sorting.unit_ids[spikes["unit_index"]])
        np.testing.assert_array_equal(events["segment_index"], spikes["segment_index"])
    metadata = json.loads((folder / "run.json").read_text())
    assert metadata["model"]["sha256"]["detector_checkpoint"]
    assert metadata["runs"][0]["clustering"]["eligible_events"] > 0
    again = vanillasort.sort(recording, model_path=tiny_model, components=2, device="cpu", verbose=False)
    np.testing.assert_array_equal(again.to_spike_vector(), spikes)
    assert torch.equal(rng_before, torch.get_rng_state())
    assert torch.get_num_threads() == threads
    with pytest.raises(ValueError, match="new or empty"):
        vanillasort.sort(recording, output_folder=folder)


def test_empty_segments_and_validation(recording, tiny_model, tmp_path):
    bundle = torch.load(tiny_model, weights_only=True)
    bundle["config"]["profiles"]["d1"].update(base_threshold=1.0, direct_keep=1.0)
    empty_model = tmp_path / "empty.pt"
    torch.save(bundle, empty_model)
    sorting = vanillasort.sort(
        recording, model_path=empty_model, components=2, device="cpu", verbose=False, output_folder=tmp_path / "empty"
    )
    assert sorting.get_num_units() == 0
    assert sorting.get_num_segments() == 2
    assert load(tmp_path / "empty/sorting").get_num_segments() == 2
    bad = NumpyRecording([np.zeros((200, 4), dtype="float32")], 20000)
    with pytest.raises(ValueError, match="30 kHz"):
        vanillasort.sort(bad)
    with pytest.raises(ValueError, match="components"):
        vanillasort.sort(recording, components=0)
    with pytest.raises(ValueError, match="fewer than K"):
        vanillasort.sort(recording, model_path=tiny_model, components=1000, device="cpu", verbose=False)
    bad = recording.select_channels(["a", "b", "c"])
    with pytest.raises(ValueError, match="at least four real"):
        vanillasort.sort(bad)


def test_geometry_and_overlap(recording, tiny_model):
    traces = recording.get_traces(segment_index=0)
    extended = NumpyRecording(
        [np.column_stack([traces, traces[:, :2]])], 30000, channel_ids=["a", "b", "c", "d", "e", "f"]
    )
    # Interleaved channel order: nearest neighbors must come from coordinates.
    extended.set_dummy_probe_from_locations(np.array([[0, 0], [0, 100], [0, 20], [0, 80], [0, 40], [0, 60]]))
    _, patches = local_neighborhoods(extended)
    assert patches[0][0].tolist() == [0, 2, 4, 5]
    assert len(patches) > 1
    sorting = vanillasort.sort(extended, model_path=tiny_model, components=2, device="cpu", verbose=False)
    assert sorting.get_num_units() > 0
    assert np.isin(sorting.get_property("main_channel_id"), extended.channel_ids).all()
    # Overlap suppresses duplicate events, but separate tetrodes keep simultaneity.
    patches = [(np.arange(4), None), (np.arange(2, 6), None), (np.arange(6, 10), None)]
    keep = deduplicate_events(np.array([100, 103, 103]), np.array([0.9, 0.8, 0.7]), np.array([0, 1, 2]), patches, 12)
    np.testing.assert_array_equal(keep, [True, False, True])


def test_checkpoint_directory_and_training(tiny_model, tmp_path):
    from vanillasort.models import VanillaDet, HuiduRep
    from vanillasort.checkpoints import load_detector, load_huidurep

    bundle = torch.load(tiny_model, weights_only=True)
    config = bundle["config"]
    for name in ("detector", "huidurep"):
        config[f"{name}_checkpoint"] = f"{name}.pt"
        state = {"module." + key: value for key, value in bundle[f"{name}_state_dict"].items()}
        torch.save({"model_state_dict": state}, tmp_path / f"{name}.pt")
    (tmp_path / "config.json").write_text(json.dumps(config))
    resolved, _ = resolve_model(model_path=tmp_path)
    detector = load_detector(resolved, "cpu")
    encoder = load_huidurep(resolved, "cpu")
    assert isinstance(detector, VanillaDet) and isinstance(encoder, HuiduRep)
    detector.train()
    loss = detector(torch.randn(2, 64, 4)).square().mean()
    loss.backward()
    assert detector.head.mlp[-1].weight.grad is not None
    for key, value in bundle["huidurep_state_dict"].items():
        torch.testing.assert_close(encoder.state_dict()[key], value)


def test_hub_cache_and_integrity(monkeypatch, tmp_path):
    import hashlib
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError
    from vanillasort import checkpoints

    # Exercise the real resolver without network access or pretrained weights.
    payload = b"tiny checkpoint cache fixture"
    manifest = dict(checkpoints.MODEL, revision="a" * 40)
    monkeypatch.setattr(checkpoints, "MODEL", manifest)
    monkeypatch.setattr(
        checkpoints, "CHECKPOINT_HASHES", {key: hashlib.sha256(payload).hexdigest() for key in manifest["files"]}
    )
    monkeypatch.setenv("VANILLASORT_MODEL_CACHE", str(tmp_path))
    calls = []

    def download(*, repo_id, filename, revision, cache_dir, token, library_name, local_files_only=False):
        assert repo_id == "Kohaku2580/VanillaSort" and revision == "a" * 40
        assert cache_dir == str(tmp_path) and token is False
        calls.append(local_files_only)
        path = tmp_path / filename
        if not path.exists():
            if local_files_only:
                raise LocalEntryNotFoundError("cache miss")
            path.write_bytes(payload)
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    config, info = checkpoints.resolve_model()
    assert calls == [True, False, True, False]
    assert info["revision"] == "a" * 40
    calls.clear()
    checkpoints.resolve_model()
    assert calls == [True, True]  # a cached run performs no remote request
    from pathlib import Path

    Path(config["detector_checkpoint"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        checkpoints.resolve_model()

    def unavailable(**kwargs):
        raise LocalEntryNotFoundError("offline and empty cache")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", unavailable)
    with pytest.raises(RuntimeError, match="model_path for offline use"):
        checkpoints.resolve_model()
