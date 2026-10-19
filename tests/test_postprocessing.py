"""Boundary policies, probe coverage, and the public fixed-K integration."""

import json

import numpy as np
import pytest
import torch
from spikeinterface.core import NumpyRecording, load

import vanillasort
from vanillasort.geometry import local_neighborhoods
from vanillasort.postprocessing import InsufficientAnchorsError, postprocess_fixed_k


def _events(units, count=24):
    labels = np.repeat(np.arange(units, dtype=np.int64), count)
    waves = np.zeros((len(labels), 72, max(1, units)), np.float32)
    waves[np.arange(len(labels)), 36, labels] = -5
    return dict(
        samples=np.arange(len(labels), dtype=np.int64) * 100,
        labels=labels,
        scores=np.linspace(0.1, 0.9, len(labels), dtype=np.float32),
        waves=waves,
    )


def test_fixed_k_empty_and_shortfall_policies():
    noise, zero, duplicate = _events(10), _events(10), _events(10)
    noise["labels"][:] = -1
    zero["waves"][:] = 0
    duplicate["waves"][:] = 0
    duplicate["waves"][:, 36, 0] = -5
    cases = [
        (_events(0), "empty_input", 0),
        (noise, "no_non_noise_events", 0),
        (zero, "no_usable_templates", 0),
        (_events(2), "insufficient_observed_clusters", 2),
        (_events(10, 1), "no_reliable_anchors", 0),
        (duplicate, "insufficient_distinct_reliable_anchors", 1),
    ]
    for data, reason, available in cases:
        with pytest.raises(InsufficientAnchorsError) as caught:
            postprocess_fixed_k(data)
        assert caught.value.audit["failure_reason"] == reason
        output, audit = postprocess_fixed_k(data, {"on_shortfall": "return_available"})
        assert audit["status"] == "partial" and not audit["exact_k"]
        assert audit["output_units"] == available
        assert np.unique(output["labels"][output["keep"]]).size == available
        assert all(n > 0 for n in audit["unit_spike_counts"])
        np.testing.assert_array_equal(output["samples"], data["samples"])
        json.dumps(audit, allow_nan=False)
    for data in (_events(10, 1), duplicate):
        _, audit = postprocess_fixed_k(data, {"anchor_policy": "supplement"})
        assert audit["status"] == "degraded" and audit["exact_k"] and audit["quality_relaxed"]
        assert audit["supplementation"]
    with pytest.raises(InsufficientAnchorsError):
        postprocess_fixed_k(zero, {"anchor_policy": "supplement"})
    with pytest.raises(InsufficientAnchorsError):
        postprocess_fixed_k(_events(2), {"anchor_policy": "supplement"})


def test_fixed_k_validation_and_nonempty_units():
    data = _events(2)
    invalid = [{"target_units": v} for v in (0, -1, True, 1.5, None)]
    invalid += [
        {"min_cosine": float("nan")},
        {"min_margin": float("inf")},
        {"max_amplitude_ratio": 0.5},
        {"anchor_policy": "invent"},
        {"target_unit": 2},
    ]
    for config in invalid:
        with pytest.raises(ValueError):
            postprocess_fixed_k(data, config)
    for bad in (
        dict(data, scores=data["scores"][:-1]),
        dict(data, waves=data["waves"][:, :-1]),
        dict(data, waves=np.full_like(data["waves"], np.nan)),
        dict(data, scores=np.full(len(data["scores"]), np.inf)),
        dict(data, labels=data["labels"].astype(float)),
        dict(data, samples=-data["samples"]),
    ):
        with pytest.raises(ValueError):
            postprocess_fixed_k(bad, {"target_units": 2})
    overflow = _events(2)
    overflow["waves"] *= np.float32(1e25)
    with pytest.raises(ValueError, match="numerical overflow"):
        postprocess_fixed_k(overflow, {"target_units": 2})
    data["labels"] = np.where(data["labels"] == 0, 7, 900)
    data["samples"][:] = 100
    before = {key: value.copy() for key, value in data.items()}
    output, audit = postprocess_fixed_k(data, {"target_units": 2})
    assert audit["unit_spike_counts"] == [1, 1]
    for unit in (7, 900):
        ids = np.flatnonzero(data["labels"] == unit)
        assert output["keep"][ids[np.argmax(data["scores"][ids])]]
    for key in data:
        np.testing.assert_array_equal(data[key], before[key])
    one = _events(1)
    one["scores"] = np.full(len(one["samples"]), 255, np.uint8)
    output, audit = postprocess_fixed_k(one, {"target_units": 1})
    assert audit["distance_margin"] == [None] and audit["exact_k"]
    assert np.all(output["quality"] == 256)
    json.dumps(audit, allow_nan=False)


def test_cover_neighborhoods_include_every_channel_once_as_owner():
    for count in (4, 5, 8, 32, 64):
        recording = NumpyRecording([np.zeros((100, count), np.float32)], 30000)
        # A two-column probe, shuffled to exercise physical vs array positions.
        locations = np.column_stack(
            (np.arange(count) // ((count + 1) // 2) * 20, np.arange(count) % ((count + 1) // 2) * 20)
        )
        order = np.random.default_rng(count).permutation(count)
        recording.set_dummy_probe_from_locations(locations[order])
        _, dense = local_neighborhoods(recording, "dense")
        _, cover = local_neighborhoods(recording, "cover")
        assert len(cover) <= len(dense)
        np.testing.assert_array_equal(np.sort(np.concatenate([owners for _, owners in cover])), np.arange(count))
        assert set(np.concatenate([channels for channels, _ in cover])) == set(range(count))
        for channels, owners in cover:
            assert len(channels) == 4 and np.all(np.diff(channels) > 0)
            assert set(owners) <= set(channels)
        if count == 4:
            np.testing.assert_array_equal(cover[0][0], dense[0][0])
        if count >= 8:
            assert len(cover) < len(dense)
    # Interleaved shanks/groups must never be combined in a detector context.
    recording = NumpyRecording([np.zeros((100, 10), np.float32)], 30000)
    recording.set_dummy_probe_from_locations(np.column_stack((np.arange(10) % 2 * 100, np.arange(10) // 2 * 20)))
    groups = np.arange(10) % 2
    recording.set_property("group", groups)
    _, cover = local_neighborhoods(recording, mode="cover")
    assert all(len(np.unique(groups[channels])) == 1 for channels, _ in cover)
    np.testing.assert_array_equal(np.sort(np.concatenate([owners for _, owners in cover])), np.arange(10))


def test_sort_fixed_k_segments_and_failure_audit(recording, tiny_model, tmp_path):
    folder = tmp_path / "fixed"
    sorting = vanillasort.sort(
        recording,
        model_path=tiny_model,
        components=2,
        target_units=1,
        postprocessing={"core_filter": False},
        device="cpu",
        verbose=False,
        output_folder=folder,
    )
    assert sorting.get_num_units() == 2  # one independent unit per segment
    for segment in range(2):
        active = [u for u in sorting.unit_ids if len(sorting.get_unit_spike_train(u, segment_index=segment))]
        assert active == [segment]
        audit = json.loads((folder / f"postprocessing_segment{segment}.json").read_text())
        assert audit["exact_k"] and audit["output_units"] == 1
    np.testing.assert_array_equal(load(folder / "sorting").to_spike_vector(), sorting.to_spike_vector())
    assert sorting.get_annotation("fixed_k_postprocessing")[0]["requested_units"] == 1
    assert np.isin(sorting.get_property("main_channel_id"), recording.channel_ids).all()
    bundle = torch.load(tiny_model, weights_only=True)
    bundle["config"]["profiles"]["d1"].update(base_threshold=1.0, direct_keep=1.0)
    empty_model = tmp_path / "empty.pt"
    torch.save(bundle, empty_model)
    empty = vanillasort.sort(
        recording,
        model_path=empty_model,
        components=2,
        target_units=10,
        postprocessing={"on_shortfall": "return_available"},
        device="cpu",
        verbose=False,
    )
    assert empty.get_num_units() == 0 and empty.get_num_segments() == 2
    failed = tmp_path / "failed"
    with pytest.raises(InsufficientAnchorsError):
        vanillasort.sort(
            recording,
            model_path=empty_model,
            components=2,
            target_units=10,
            device="cpu",
            verbose=False,
            output_folder=failed,
        )
    assert json.loads((failed / "postprocessing_segment0.json").read_text())["failure_reason"] == "empty_input"
    assert not (failed / "sorting").exists()
