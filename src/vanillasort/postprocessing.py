"""Utilities for validated fixed-K merging of local spike clusters.

One segment, 30 kHz, with normalized 72-sample waveforms on all channels.
Label -1 denotes input noise. Only events with output['keep'] are emitted.
The default raises on insufficient distinct reliable anchors. Set
on_shortfall='return_available' to return fewer units, or explicitly opt into
anchor_policy='supplement' to reuse weak/duplicate observed clusters.
Neither policy invents events, empty units, or new clusters.
"""

from collections.abc import Mapping
import time

import numpy as np
from threadpoolctl import threadpool_limits


def _crops(waves, shifts):
    rows = np.arange(len(waves))[:, None]
    times = 6 + np.asarray(shifts)[:, None] + np.arange(60)
    values = np.asarray(waves[rows, times], np.float32).copy()
    values -= values.mean(axis=1, keepdims=True)
    return values


def _cluster_templates(d):
    units = np.unique(d["labels"])
    indices = np.searchsorted(units, d["labels"])
    templates, counts, centers = [], [], []
    for unit in units:
        ids = np.flatnonzero(d["labels"] == unit)
        if len(ids) > 500:
            ids = ids[np.linspace(0, len(ids) - 1, 500).astype(int)]
        median = np.median(d["waves"][ids], axis=0)
        peak = 30 + np.argmax(np.max(np.abs(median[30:43]), axis=1))
        shift = int(np.clip(peak - 36, -4, 4))
        t = median[6 + shift : 66 + shift].copy()
        t -= t.mean(axis=0, keepdims=True)
        templates.append(t)
        counts.append(np.sum(d["labels"] == unit))
        centers.append(shift)
    return np.asarray(templates, np.float32), np.asarray(counts), np.asarray(centers), indices, units


def _characteristics(d):
    templates, counts, centers, indices, local_units = _cluster_templates(d)
    peaks, offsets = [], []
    for i in range(len(templates)):
        ids = np.flatnonzero(indices == i)
        if len(ids) > 500:
            ids = ids[np.linspace(0, len(ids) - 1, 500).astype(int)]
        median = np.median(d["waves"][ids], axis=0)
        peaks.append(float(np.abs(median).max()))
        offsets.append(int(np.unravel_index(np.abs(median).argmax(), median.shape)[0]) - 36)
    similarity = np.eye(len(templates))
    for shift in range(-3, 4):
        left = templates[:, max(0, shift) : min(60, 60 + shift)].reshape(len(templates), -1)
        right = templates[:, max(0, -shift) : min(60, 60 - shift)].reshape(len(templates), -1)
        score = (
            left
            @ right.T
            / np.maximum(np.linalg.norm(left, axis=1)[:, None] * np.linalg.norm(right, axis=1)[None, :], 1e-12)
        )
        similarity = np.maximum(similarity, score)
    norms = np.linalg.norm(templates.reshape(len(templates), -1), axis=1)
    logratio = np.log(np.maximum(norms, 1e-8))[:, None] - np.log(np.maximum(norms, 1e-8))[None, :]
    return (
        templates,
        counts,
        centers,
        indices,
        local_units,
        np.asarray(peaks),
        np.asarray(offsets),
        similarity,
        logratio,
    )


def _deduplicate_units(samples, scores, labels, radius=12):
    # Greedy score priority; comparisons are strictly WITHIN assigned unit.
    keep = np.zeros(len(samples), dtype=bool)
    accepted = set()
    for index in np.argsort(-scores, kind="stable"):
        label, sample = int(labels[index]), int(samples[index])
        if any((label, t) in accepted for t in range(sample - radius, sample + radius + 1)):
            continue
        accepted.add((label, sample))
        keep[index] = True
    return keep


class InsufficientAnchorsError(ValueError):
    """No exact-K result under the chosen policy; .audit contains diagnostics."""

    def __init__(self, audit):
        self.audit = audit
        super().__init__(
            f"{audit['failure_reason']}: requested {audit['requested_units']} units; "
            f"observed={audit['observed_local_clusters']}, "
            f"usable={audit['usable_local_clusters']}, "
            f"distinct reliable anchors={audit['distinct_reliable_anchors']}, "
            f"available under policy={audit['available_units']}. "
            "Use on_shortfall='return_available' for an explicitly partial result. "
            "anchor_policy='supplement' explicitly permits weak/duplicate clusters."
        )


def jsonable(value):
    """Convert audits to strict JSON types (single-anchor margins are null)."""
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def _config(config):
    defaults = dict(
        target_units=10,
        core_filter=True,
        anchor_only=False,
        rank_quality="coherence",
        anchor_duplicate_cosine=0.995,
        amp_weight=0.25,
        min_fragment_coherence=0.5,
        min_cosine=0.9,
        max_amplitude_ratio=1.5,
        min_margin=0.03,
        anchor_policy="reliable_only",
        on_shortfall="raise",
    )
    if config is not None and not isinstance(config, Mapping):
        raise ValueError("config must be a mapping")
    supplied = dict(config or {})
    unknown = supplied.keys() - defaults.keys()
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown, key=str)}")
    defaults.update(supplied)
    k = defaults["target_units"]
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or not 1 <= k <= np.iinfo(np.int32).max:
        raise ValueError("target_units must be a positive integer within int32 range")
    defaults["target_units"] = int(k)
    for key in ("core_filter", "anchor_only"):
        if not isinstance(defaults[key], (bool, np.bool_)):
            raise ValueError(f"{key} must be boolean")
        defaults[key] = bool(defaults[key])
    choices = {
        "rank_quality": ("none", "coherence"),
        "anchor_policy": ("reliable_only", "supplement"),
        "on_shortfall": ("raise", "return_available"),
    }
    for key, allowed in choices.items():
        if not isinstance(defaults[key], str) or defaults[key] not in allowed:
            raise ValueError(f"{key} must be one of {allowed}")
    bounds = {
        "anchor_duplicate_cosine": (-1, None),
        "amp_weight": (0, None),
        "min_fragment_coherence": (0, 1),
        "min_cosine": (-1, 1),
        "max_amplitude_ratio": (1, None),
        "min_margin": (0, None),
    }
    for key, (lo, hi) in bounds.items():
        value = defaults[key]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"{key} must be a finite number")
        try:
            value = float(value)
        except OverflowError as exc:
            raise ValueError(f"{key} is outside the supported numeric range") from exc
        if not np.isfinite(value) or value < lo or (hi is not None and value > hi):
            raise ValueError(f"Invalid {key}: {value}; expected >= {lo}" + (f" and <= {hi}" if hi is not None else ""))
        defaults[key] = value
    return defaults


def _data(data):

    if not isinstance(data, Mapping):
        raise ValueError("data must be a mapping")
    required = ("samples", "labels", "scores", "waves")

    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Missing input fields: {missing}")

    d = {key: np.asarray(data[key]) for key in required}
    for key in ("samples", "labels", "scores"):
        if d[key].ndim != 1:
            raise ValueError(f"{key} must be one-dimensional")

    n = len(d["samples"])
    if len(d["labels"]) != n or len(d["scores"]) != n:
        raise ValueError("samples, labels, and scores must have the same length")

    for key, minimum in (("samples", 0), ("labels", -1)):
        values = d[key]
        if values.dtype.kind not in "iu":
            raise ValueError(f"{key} must contain integers (not booleans)")
        if len(values) and (int(values.min()) < minimum or int(values.max()) > np.iinfo(np.int64).max):
            raise ValueError(f"{key} must be in [{minimum}, int64 max]; only label -1 denotes noise")
        d[key] = values.astype(np.int64, copy=False)
    scores = d["scores"]

    if scores.dtype.kind not in "fiu" or not np.isfinite(scores).all():
        raise ValueError("scores must be finite real numbers")

    # Preserve the normal float32/float64 path; avoid integer +1 overflow.
    if scores.dtype.kind != "f":
        scores = scores.astype(np.float64)
    elif scores.dtype.itemsize < 4:
        scores = scores.astype(np.float32)
    d["scores"] = scores
    
    waves = d["waves"]
    if waves.ndim != 3 or waves.shape[0] != n or waves.shape[1] != 72 or waves.shape[2] < 1:
        raise ValueError("waves must have shape (number_of_events, 72, number_of_channels >= 1)")
    if waves.dtype.kind != "f" or waves.dtype.itemsize not in (4, 8):
        raise ValueError("waves must be float32 or float64 normalized waveforms")
    # Bound temporary memory for cached/memory-mapped multichannel recordings.
    for start in range(0, n, 2048):
        if not np.isfinite(waves[start : start + 2048]).all():
            raise ValueError("waves contains NaN or infinity")
    return d


def _coherence(d, templates, centers, indices, usable, config):
    coherence = np.ones(len(templates))
    coherence[~usable] = 0
    if config["rank_quality"] == "none":
        return coherence
    for i in np.flatnonzero(usable):
        ids = np.flatnonzero(indices == i)
        if len(ids) > 500:
            ids = ids[np.linspace(0, len(ids) - 1, 500).astype(int)]
        values = _crops(d["waves"][ids], np.full(len(ids), centers[i])).reshape(len(ids), -1)
        template = templates[i].ravel()
        energy = np.sum(values * values, axis=1)
        norm = float(template @ template)
        dot = values @ template
        signal_energy = np.maximum(energy - values.shape[1], 0.25 * norm)
        corrected_cosine = dot / np.maximum(np.sqrt(signal_energy * norm), 1e-8)
        amplitude = dot / max(norm, 1e-8)
        coherence[i] = ((corrected_cosine >= 0.9) & (amplitude >= 2 / 3) & (amplitude <= 1.5)).mean()
    return coherence


def _run(d, config, start):
    k = config["target_units"]
    n = len(d["samples"])
    valid = np.flatnonzero(d["labels"] >= 0)
    # Do not copy a full waveform cache on the ordinary all-valid path.
    working = d if len(valid) == n else {key: value[valid] for key, value in d.items()}
    units = np.unique(working["labels"])
    anchors, supplemented, supplementation = [], [], []
    diagnostic = dict(
        version=2,
        config=config,
        requested_units=k,
        input_events=n,
        input_noise_events=n - len(valid),
        observed_local_clusters=len(units),
        usable_local_clusters=0,
        reliable_local_clusters=0,
        distinct_reliable_anchors=0,
        available_units=0,
        failure_reason=None,
        quality_relaxed=not config["core_filter"],
        supplemented_anchor_local_units=supplemented,
        supplementation=supplementation,
        degenerate_local_units=[],
        local_unit_ids=units,
        anchor_local_units=[],
        anchor_event_counts=[],
        reliable_local_units=[],
        accepted_local_units=[],
        coherent_event_fraction=[],
        anchor_rank_score=[],
        cluster_to_unit=[],
        chosen_cosine=[],
        chosen_log_amplitude_ratio=[],
        distance_margin=[],
        peak_offset=[],
        full_template_peak=[],
    )
    if len(units):
        templates, counts, centers, indices, units, peaks, offsets, similarity, logratio = _characteristics(working)
        norms = np.linalg.norm(templates.reshape(len(units), -1), axis=1)
        if not all(np.isfinite(value).all() for value in (templates, peaks, similarity, logratio, norms)):
            raise FloatingPointError("Non-finite template statistics")
        usable = norms > 1e-8
        core = usable & (counts >= 20) & (np.abs(offsets) <= 8) & (peaks >= 2.5)
        if not config["core_filter"]:
            core = usable.copy()
        coherence = _coherence(working, templates, centers, indices, usable, config)
        rank_score = counts * coherence
        rank = np.argsort(-rank_score, kind="stable")
        candidates = rank[core[rank]].tolist()
        duplicate_skipped = set()
        for i in candidates:
            if anchors and np.any(
                (similarity[i, anchors] >= config["anchor_duplicate_cosine"])
                & (np.abs(logratio[i, anchors]) <= np.log(1.15))
            ):
                duplicate_skipped.add(i)
                continue
            anchors.append(i)
            if len(anchors) == k:
                break
        distinct_reliable = len(anchors)
        if config["anchor_policy"] == "supplement" and len(anchors) < k:
            for i in candidates + rank[usable[rank]].tolist():
                if i in anchors:
                    continue
                anchors.append(i)
                supplemented.append(int(units[i]))
                supplementation.append(
                    {
                        "local_unit": int(units[i]),
                        "reason": "core_filter_failed" if not core[i] else "duplicate_template",
                    }
                )
                if len(anchors) == k:
                    break
        diagnostic.update(
            usable_local_clusters=int(usable.sum()),
            reliable_local_clusters=int(core.sum()),
            distinct_reliable_anchors=distinct_reliable,
            available_units=len(anchors),
            quality_relaxed=bool(supplemented) or not config["core_filter"],
            degenerate_local_units=units[~usable],
            duplicate_skipped_local_units=units[sorted(duplicate_skipped)],
            anchor_local_units=units[anchors],
            anchor_event_counts=counts[anchors],
            reliable_local_units=units[core],
            coherent_event_fraction=coherence,
            anchor_rank_score=rank_score,
            peak_offset=offsets,
            full_template_peak=peaks,
        )
    if len(anchors) < k:
        if n == 0:
            reason = "empty_input"
        elif not len(units):
            reason = "no_non_noise_events"
        elif not diagnostic["usable_local_clusters"]:
            reason = "no_usable_templates"
        elif len(units) < k:
            reason = "insufficient_observed_clusters"
        elif config["anchor_policy"] == "reliable_only":
            reason = "no_reliable_anchors" if not anchors else "insufficient_distinct_reliable_anchors"
        else:
            reason = "insufficient_usable_clusters"
        diagnostic.update(failure_reason=reason, exact_k=False)
        if config["on_shortfall"] == "raise":
            diagnostic.update(status="error", postprocess_seconds=time.perf_counter() - start)
            raise InsufficientAnchorsError(jsonable(diagnostic))

    labels = np.full(n, -1, np.int32)
    noise = np.ones(n, bool)
    quality = d["scores"].copy()
    anchor_events = np.zeros(n, bool)
    if anchors:
        anchors = np.asarray(anchors, np.int64)
        actual_k = len(anchors)
        distance = 1 - similarity[:, anchors] + config["amp_weight"] * np.abs(logratio[:, anchors])
        if not np.isfinite(distance).all():
            raise FloatingPointError("Non-finite anchor distances")
        order = np.argsort(distance, axis=1)
        mapping = order[:, 0].astype(np.int32)
        row = np.arange(len(units))
        chosen_cosine = similarity[row, anchors[mapping]]
        chosen_logratio = logratio[row, anchors[mapping]]
        margins = distance[row, order[:, 1]] - distance[row, order[:, 0]] if actual_k > 1 else None
        margin_ok = margins >= config["min_margin"] if margins is not None else np.ones(len(units), bool)
        accepted = (
            core
            & (coherence >= config["min_fragment_coherence"])
            & (chosen_cosine >= config["min_cosine"])
            & (np.abs(chosen_logratio) <= np.log(config["max_amplitude_ratio"]))
            & margin_ok
        )
        if config["anchor_only"]:
            accepted[:] = False
        mapping[anchors] = np.arange(actual_k)
        accepted[anchors] = True
        labels[valid] = mapping[indices]
        noise[valid] = ~accepted[indices]
        anchor_events[valid] = np.isin(indices, anchors)
        quality[anchor_events] += 1
        diagnostic.update(
            cluster_to_unit=mapping,
            accepted_local_units=units[accepted],
            chosen_cosine=chosen_cosine,
            chosen_log_amplitude_ratio=chosen_logratio,
            distance_margin=margins if margins is not None else [None] * len(units),
        )
    else:
        diagnostic.update(cluster_to_unit=[-1] * len(units), distance_margin=[None] * len(units))
    keep = np.zeros(n, bool)
    ids = np.flatnonzero(~noise)
    if len(ids):
        keep[ids] = _deduplicate_units(d["samples"][ids], quality[ids], labels[ids])
    output_units, sizes = np.unique(labels[keep], return_counts=True)
    # Every selected observed anchor supplies at least one surviving event;
    # deduplication is per unit, including when different units coincide in time.
    if not np.array_equal(output_units, np.arange(len(anchors))) or not np.all(sizes > 0):
        raise RuntimeError("Postprocessing invariant failed: a selected unit became empty")
    diagnostic.update(
        status="partial" if len(anchors) < k else ("degraded" if diagnostic["quality_relaxed"] else "ok"),
        output_units=len(output_units),
        unit_spike_counts=sizes,
        output_spikes=int(keep.sum()),
        exact_k=len(output_units) == k,
        rejected_events=int(noise.sum()),
        dedup_removed=int((~noise).sum() - keep.sum()),
        preserved_anchor_events=int(anchor_events.sum()),
        kept_anchor_events=int((anchor_events & keep).sum()),
        changed_timestamps=0,
        gt_used=False,
        postprocess_seconds=time.perf_counter() - start,
    )
    return dict(samples=d["samples"].copy(), labels=labels, keep=keep, noise=noise, quality=quality), jsonable(
        diagnostic
    )


def postprocess_fixed_k(data, config=None):
    """Return event arrays and a JSON-safe audit, or a descriptive ValueError.

    'reliable' keeps v1's count >= 20, peak >= 2.5, and offset <= 8 criteria.
    Coherence ranks anchors; min_fragment_coherence filters merge candidates.
    core_filter=False is an explicit relaxation of the core criteria.
    With no competing anchor (K=1), the margin test is inapplicable.
    """
    start = time.perf_counter()
    config = _config(config)
    data = _data(data)
    try:
        with threadpool_limits(limits=1), np.errstate(over="raise", invalid="raise", divide="raise"):
            return _run(data, config, start)
    except FloatingPointError as exc:
        raise ValueError(
            "Waveforms/scores or configuration caused numerical overflow; check normalization and scale"
        ) from exc


def extract_postprocessing_waveforms(recording, segment_index, samples, filter_config):
    """Extract MAD-scaled 72-sample waveforms on a common physical-channel basis.

    Filter complete channels, four at a time, preserving full-segment MAD and
    filter boundaries. This avoids retaining the full multichannel recording
    alongside the waveform array. No detector or embedding model is run here.
    """
    import psutil

    from .ops import robust_location_scale_time_channel
    from .pipeline import filtered_signal

    samples = np.asarray(samples)
    if samples.ndim != 1 or samples.dtype.kind not in "iu":
        raise ValueError("samples must be a one-dimensional integer array")
    frames = recording.get_num_samples(segment_index=segment_index)
    if len(samples) and (int(samples.min()) < 0 or int(samples.max()) >= frames):
        raise ValueError("Postprocessing samples are outside the recording segment")
    channels = recording.get_num_channels()
    if not len(samples):
        return np.empty((0, 72, channels), dtype=np.float32)
    required = len(samples) * 72 * (channels * 4 + 16) + 12 * frames * min(4, channels) * 4 + 512 * 2**20
    if psutil.virtual_memory().available < required:
        raise MemoryError(f"Fixed-K waveform extraction needs approximately {required / 2**30:.2f} GiB available RAM")
    indices = samples.astype(np.int64)[:, None] + np.arange(-36, 36)
    valid = (indices >= 0) & (indices < frames)
    indices = np.clip(indices, 0, frames - 1)
    waves = np.empty((len(samples), 72, channels), dtype=np.float32)
    channel_ids = np.asarray(recording.get_channel_ids())
    for first in range(0, channels, 4):
        last = min(channels, first + 4)
        raw = np.empty((frames, last - first), dtype=np.float32)
        for start in range(0, frames, 300_000):
            stop = min(frames, start + 300_000)
            raw[start:stop] = recording.get_traces(
                segment_index=segment_index,
                start_frame=start,
                end_frame=stop,
                channel_ids=channel_ids[first:last],
            )
        if recording.has_scaleable_traces():
            raw *= recording.get_channel_gains()[first:last]
            raw += recording.get_channel_offsets()[first:last]
        physical = filtered_signal(raw, recording.get_sampling_frequency(), filter_config)
        _, scale = robust_location_scale_time_channel(physical)
        waves[:, :, first:last] = physical[indices] / scale
        del raw, physical
    waves[~valid] = 0
    waves -= waves.mean(axis=1, keepdims=True)
    return waves
