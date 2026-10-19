"""Detector threshold overrides and optional user-defined event selection."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np


@dataclass(frozen=True)
class GateContext:
    """Input to ``gate_fn(context)`` for one segment/neighborhood.

    Arrays are read-only copies on CPU. ``samples`` (N,) are zero-based indices
    within the segment; ``scores`` (N,) have already passed ``threshold``.
    ``snrs`` (N, 4) contains negative peak amplitudes on the canonical filtered,
    median/MAD-normalized signal, measured within ``event_snr_radius`` samples.
    Its columns follow ``channel_ids`` and the original ``channel_locations``
    (4, 2), in micrometres. No SNR or ownership selection has run yet.

    Return a boolean mask of shape (N,). True retains an event; a custom gate
    cannot add events, change timestamps, or bypass the score threshold.
    The callback is skipped if no events pass that threshold. Exceptions from
    the callback propagate to the caller. Keep the callback code and any closed-
    over parameters alongside your experiment; only its name is logged.
    """

    samples: np.ndarray
    scores: np.ndarray
    snrs: np.ndarray
    fs_hz: float
    channel_ids: np.ndarray
    channel_locations: np.ndarray
    segment_index: int
    neighborhood_index: int
    profile: str
    event_snr_radius: int


def resolve_gate(
    config,
    profile,
    *,
    enable_gate=True,
    threshold=None,
    gate_fn=None,
    direct_keep=None,
    primary_snr=None,
    secondary_snr=None,
    min_secondary_channels=None,
):
    """Validate options and return JSON-safe settings without changing config."""
    if profile not in config["profiles"]:
        raise ValueError(f"Unknown profile {profile!r}")
    if not isinstance(enable_gate, (bool, np.bool_)):
        raise ValueError("enable_gate must be a boolean")
    if gate_fn is not None:
        if not callable(gate_fn):
            raise TypeError("gate_fn must be callable or None")
        if not enable_gate:
            raise ValueError("gate_fn requires enable_gate=True; omit gate_fn to disable gating")
    p = config["profiles"][profile]
    settings = {}
    for name, value, default, upper in (
        ("threshold", threshold, p["base_threshold"], 1),
        ("direct_keep", direct_keep, p["direct_keep"], 1),
        ("primary_snr", primary_snr, config["primary_snr"], np.inf),
        ("secondary_snr", secondary_snr, config["secondary_snr"], np.inf),
    ):
        value = default if value is None else value
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Real)
            or not np.isfinite(value)
            or not 0 <= value <= upper
        ):
            bounds = "in [0, 1]" if upper == 1 else ">= 0"
            raise ValueError(f"{name} must be a finite number {bounds}")
        settings[name] = float(value)
    count = config["min_secondary_channels"] if min_secondary_channels is None else min_secondary_channels
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or not 1 <= count <= 3:
        raise ValueError("min_secondary_channels must be an integer in [1, 3]")
    settings.update(
        enable_gate=bool(enable_gate),
        mode="custom" if gate_fn is not None else "snr" if enable_gate else "off",
        min_secondary_channels=int(count),
        callback=(
            f"{getattr(gate_fn, '__module__', type(gate_fn).__module__)}."
            f"{getattr(gate_fn, '__qualname__', type(gate_fn).__qualname__)}"
            if gate_fn is not None
            else None
        ),
    )
    return settings


def _read_only_copy(array):
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def select_candidates(samples, scores, snrs, settings, gate_fn=None, **context_fields):
    """Return indices into the original candidate arrays, plus gate diagnostics."""
    eligible = np.flatnonzero(scores >= settings["threshold"])
    positions = eligible
    callback_called = False
    if settings["mode"] == "snr":
        # Keep the published numerical kernel unchanged, including boundaries.
        from .ops import filter_scored_predictions_with_ranked_event_snr_gate

        pred, _ = filter_scored_predictions_with_ranked_event_snr_gate(
            samples,
            scores,
            snrs,
            settings["threshold"],
            settings["direct_keep"],
            settings["primary_snr"],
            settings["secondary_snr"],
            settings["min_secondary_channels"],
        )
        positions = np.searchsorted(samples, pred)
    elif settings["mode"] == "custom" and len(eligible):
        context = GateContext(
            samples=_read_only_copy(samples[eligible]),
            scores=_read_only_copy(scores[eligible]),
            snrs=_read_only_copy(snrs[eligible]),
            **{
                key: _read_only_copy(value) if key in ("channel_ids", "channel_locations") else value
                for key, value in context_fields.items()
            },
        )
        keep = np.asarray(gate_fn(context))
        if keep.dtype != np.dtype(bool) or keep.shape != (len(eligible),):
            raise ValueError(
                f"gate_fn must return a boolean mask of shape ({len(eligible)},); "
                f"got shape {keep.shape}, dtype {keep.dtype}"
            )
        positions = eligible[keep]
        callback_called = True
    return positions, dict(
        **settings,
        candidate_events=len(samples),
        below_threshold=len(samples) - len(eligible),
        threshold_events=len(eligible),
        rejected_events=len(eligible) - len(positions),
        accepted_events=len(positions),
        callback_called=callback_called,
    )
