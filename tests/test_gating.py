import copy
import json
import sys

import numpy as np
import pytest
import torch

import vanillasort
from vanillasort.checkpoints import load_config
from vanillasort.gating import resolve_gate, select_candidates
from vanillasort.ops import filter_scored_predictions_with_ranked_event_snr_gate as original_gate


def test_gate_defaults_overrides_and_validation():
    config = load_config()
    before = copy.deepcopy(config)
    samples = np.arange(20, 200, 20, dtype=np.int64)
    scores = np.array([0.02, 0.038, 0.039, 0.04, 0.041, 0.044, 0.045, 0.1, 1], dtype=np.float32)
    snrs = np.array([[3, 2, 1, 0], [1, 0, 0, 0], [3, 2, 0, 0]] * 3, dtype=np.float32)
    for profile in ("d1", "canonical"):
        p = config["profiles"][profile]
        expected, rejected = original_gate(samples, scores, snrs, p["base_threshold"], p["direct_keep"], 3, 2, 1)
        positions, audit = select_candidates(samples, scores, snrs, resolve_gate(config, profile))
        np.testing.assert_array_equal(samples[positions], expected)
        assert audit["rejected_events"] == rejected
        # Turning the gate off is equivalent to the previous config workaround.
        expected, _ = original_gate(samples, scores, snrs, p["base_threshold"], p["base_threshold"], 3, 2, 1)
        positions, audit = select_candidates(samples, scores, snrs, resolve_gate(config, profile, enable_gate=False))
        np.testing.assert_array_equal(samples[positions], expected)
        assert audit["rejected_events"] == 0 and audit["mode"] == "off"
    settings = resolve_gate(
        config,
        "d1",
        threshold=np.float64(0.04),
        direct_keep=0.8,
        primary_snr=2,
        secondary_snr=1,
        min_secondary_channels=np.int64(2),
    )
    expected, rejected = original_gate(samples, scores, snrs, 0.04, 0.8, 2, 1, 2)
    positions, audit = select_candidates(samples, scores, snrs, settings)
    np.testing.assert_array_equal(samples[positions], expected)
    assert audit["rejected_events"] == rejected
    # Threshold and direct_keep remain independent, including threshold > direct_keep.
    positions, _ = select_candidates(samples, scores, snrs, resolve_gate(config, "d1", threshold=0.139))
    np.testing.assert_array_equal(samples[positions], samples[scores >= 0.139])
    for kwargs in (
        {"threshold": -0.1},
        {"threshold": 1.1},
        {"threshold": np.nan},
        {"threshold": True},
        {"threshold": "0.1"},
        {"direct_keep": np.inf},
        {"primary_snr": -1},
        {"secondary_snr": np.nan},
        {"min_secondary_channels": 0},
        {"min_secondary_channels": 4},
        {"min_secondary_channels": 1.5},
        {"min_secondary_channels": True},
        {"enable_gate": 0},
        {"gate_fn": 3},
        {"enable_gate": False, "gate_fn": lambda context: []},
    ):
        with pytest.raises((ValueError, TypeError)):
            resolve_gate(config, "d1", **kwargs)
    json.dumps(settings, allow_nan=False)
    assert config == before


def test_custom_gate_contract_and_ownership(monkeypatch):
    from vanillasort import pipeline

    samples = np.array([40, 80, 120, 160, 200], dtype=np.int64)
    scores = np.array([0.02, 0.04, 0.05, 0.8, 0.6], dtype=np.float32)
    snrs = np.array([[0, 0, 0, 0], [3, 5, 1, 0], [3, 1, 0, 0], [5, 1, 0, 0], [4, 1, 0, 0]], np.float32)
    monkeypatch.setattr(pipeline, "load_detector", lambda *args: None)
    monkeypatch.setattr(pipeline, "infer_candidates", lambda *args, **kwargs: (samples, scores, {}))
    monkeypatch.setattr(pipeline.ops, "estimate_candidate_event_snrs_by_channel", lambda *args: snrs)
    coords = np.array([[10, 10], [30, 10], [10, 30], [30, 30]], np.float32)
    raw = np.random.default_rng(4).normal(size=(240, 4)).astype(np.float32)
    args = (raw, coords, 30000, load_config(), torch.device("cpu"), 2)
    options = dict(
        detect_only=True,
        verbose=False,
        owner_channels=[0],
        channel_ids=["a", "b", "c", "d"],
        segment_index=2,
        neighborhood_index=3,
    )
    calls = []

    def custom(context):
        calls.append(context)
        np.testing.assert_array_equal(context.samples, samples[1:])
        np.testing.assert_array_equal(context.channel_locations, coords)
        assert context.channel_ids.tolist() == ["a", "b", "c", "d"]
        assert (context.segment_index, context.neighborhood_index, context.fs_hz) == (2, 3, 30000)
        assert context.profile == "d1" and context.event_snr_radius == 12
        for array in (context.samples, context.scores, context.snrs, context.channel_ids, context.channel_locations):
            assert not array.flags.writeable
        with pytest.raises(ValueError):
            context.scores[0] = 0
        # Even if deliberately made writable, the copy cannot corrupt pipeline data.
        context.samples.setflags(write=True)
        context.samples[:] = -1
        return [True, True, True, False]

    arrays, audit = pipeline.run_pipeline(*args, **options, gate_fn=custom)
    np.testing.assert_array_equal(arrays["detection_samples"], [120, 160])
    np.testing.assert_array_equal(arrays["detection_scores"], scores[[2, 3]])
    np.testing.assert_array_equal(arrays["detection_snrs"], snrs[[2, 3]])
    assert len(calls) == 1
    assert audit["gating"]["below_threshold"] == 1
    assert audit["gate_rejected"] == 1 and audit["ownership_removed"] == 1
    assert audit["gating"]["accepted_events"] == 3
    assert audit["gating"]["mode"] == "custom" and audit["gating"]["callback_called"]
    # No direct_keep bypass for a custom gate; high scores can also be rejected.
    assert 200 not in arrays["detection_samples"]
    empty, audit = pipeline.run_pipeline(*args, **options, gate_fn=custom, threshold=1)
    assert not len(empty["detection_samples"]) and len(calls) == 1
    assert not audit["gating"]["callback_called"]
    off, audit = pipeline.run_pipeline(*args, **options, enable_gate=False)
    np.testing.assert_array_equal(off["detection_samples"], [120, 160, 200])
    assert audit["gate_rejected"] == 0 and audit["ownership_removed"] == 1
    for invalid in (np.ones(4, dtype=int), np.ones((4, 1), dtype=bool), np.ones(3, dtype=bool), None):
        with pytest.raises(ValueError, match="boolean mask"):
            pipeline.run_pipeline(*args, **options, gate_fn=lambda context: invalid)

    def broken(context):
        raise RuntimeError("user gate failed")

    with pytest.raises(RuntimeError, match="user gate failed"):
        pipeline.run_pipeline(*args, **options, gate_fn=broken)


def test_api_gate_callback_roundtrip(recording, tiny_model, tmp_path, monkeypatch):
    from vanillasort import checkpoints

    config, model_info = checkpoints.resolve_model(model_path=tiny_model)
    before = copy.deepcopy(config)
    monkeypatch.setattr(checkpoints, "resolve_model", lambda *args: (config, model_info))
    options = dict(model_path=tiny_model, components=2, device="cpu", verbose=False)
    baseline = tmp_path / "baseline"
    original = vanillasort.sort(recording, output_folder=baseline, **options)
    # Default tiny model has direct_keep == threshold; disabling gating is identical.
    disabled = vanillasort.sort(recording, enable_gate=False, **options)
    np.testing.assert_array_equal(original.to_spike_vector(), disabled.to_spike_vector())
    calls = []

    def custom(context):
        calls.append((context.segment_index, context.neighborhood_index))
        return (context.samples % 2 == 0) & (context.segment_index == 0)

    folder = tmp_path / "custom"
    result = vanillasort.sort(recording, gate_fn=custom, threshold=0, output_folder=folder, **options)
    assert calls == [(0, 0), (1, 0)]
    assert result.get_num_segments() == 2
    assert result.get_num_units() > 0
    assert not np.any(result.to_spike_vector()["segment_index"] == 1)
    for segment in (0, 1):
        with (
            np.load(baseline / f"segment{segment}_patch0.npz") as old,
            np.load(folder / f"segment{segment}_patch0.npz") as new,
        ):
            samples = old["detection_samples"]
            keep = (samples % 2 == 0) & (segment == 0)
            np.testing.assert_array_equal(new["detection_samples"], samples[keep])
            np.testing.assert_array_equal(new["detection_scores"], old["detection_scores"][keep])
    metadata = json.loads((folder / "run.json").read_text())
    assert metadata["gating"]["threshold"] == 0
    assert metadata["gating"]["callback"].endswith(".<locals>.custom")
    assert all(run["gating"]["mode"] == "custom" for run in metadata["runs"])
    assert result.get_annotation("detection_gate") == metadata["gating"]
    assert metadata["config"] == before and config == before
    for kwargs in ({"enable_gate": False, "gate_fn": custom}, {"threshold": np.nan}):
        with pytest.raises(ValueError):
            vanillasort.sort(recording, output_folder=tmp_path / "invalid", **kwargs, **options)
        assert not (tmp_path / "invalid").exists()


def test_cli_gate_options(recording, tiny_model, tmp_path, monkeypatch):
    from vanillasort.cli import main

    path = tmp_path / "input.npz"
    np.savez(path, traces=recording.get_traces(segment_index=0), coords=recording.get_channel_locations())
    output = tmp_path / "cli"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vanillasort",
            "--input",
            str(path),
            "--model-path",
            str(tiny_model),
            "--device",
            "cpu",
            "--detect-only",
            "--no-gate",
            "--threshold",
            "1",
            "--direct-keep",
            "0.8",
            "--primary-snr",
            "4",
            "--secondary-snr",
            "2.5",
            "--min-secondary-channels",
            "2",
            "--output",
            str(output),
        ],
    )
    threads = torch.get_num_threads()
    try:
        with torch.random.fork_rng():
            assert main() == 0
    finally:
        torch.set_num_threads(threads)
    audit = json.loads((output / "run.json").read_text())
    settings = audit["gating"]
    assert settings["mode"] == "off" and settings["threshold"] == 1
    assert settings["direct_keep"] == 0.8 and settings["primary_snr"] == 4
    assert settings["secondary_snr"] == 2.5 and settings["min_secondary_channels"] == 2
    assert audit["config"]["profiles"]["d1"]["base_threshold"] == 0.0001
    with np.load(output / "events.npz") as events:
        assert events["detection_samples"].size == 0
