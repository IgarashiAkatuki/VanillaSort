"""Frozen numerical kernels extracted unchanged from the verified experiment code.
See MANIFEST.json for source identities. No original project imports are needed.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Sequence
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import maximum_filter1d
from scipy.linalg import cholesky, solve_triangular, helmert
from sklearn.preprocessing import StandardScaler


# Source: SpikeDetector/detector/evaluate_hybrid_janelia.py :: robust_location_scale_time_channel
def robust_location_scale_time_channel(
    x: np.ndarray, eps: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-channel median and robust scale for a [time, channel] array."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected [time, channel], got {x.shape}")
    med = np.median(x, axis=0, keepdims=True)
    mad = np.median(np.abs(x - med), axis=0, keepdims=True) + eps
    scale = 1.4826 * mad + eps
    if not np.isfinite(med).all() or not np.isfinite(scale).all():
        raise FloatingPointError("Robust location/scale produced NaN/Inf")
    return med.astype(np.float32, copy=False), scale.astype(np.float32, copy=False)

# Source: SpikeDetector/detector/evaluate_hybrid_janelia.py :: robust_normalize_time_channel
def robust_normalize_time_channel(
    x: np.ndarray,
    eps: float = 1e-6,
    *,
    location: np.ndarray | None = None,
    scale: np.ndarray | None = None,
) -> np.ndarray:
    """Median/MAD normalize [time, channel], optionally with fixed statistics."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected [time, channel], got {x.shape}")
    if (location is None) != (scale is None):
        raise ValueError("location and scale must either both be supplied or both omitted")
    if location is None:
        location, scale = robust_location_scale_time_channel(x, eps=eps)
    else:
        location = np.asarray(location, dtype=np.float32)
        scale = np.asarray(scale, dtype=np.float32)
        expected = (1, x.shape[1])
        if location.shape != expected or scale.shape != expected:
            raise ValueError(
                f"Expected location/scale shape {expected}, got "
                f"{location.shape} and {scale.shape}"
            )
        if not np.isfinite(location).all() or not np.isfinite(scale).all():
            raise FloatingPointError("Fixed robust location/scale contains NaN/Inf")
        if bool((scale <= 0.0).any()):
            raise ValueError("Fixed robust scale must be positive")
    out = (x - location) / scale
    if not np.isfinite(out).all():
        raise FloatingPointError("Robust normalization produced NaN/Inf")
    return np.ascontiguousarray(out, dtype=np.float32)

# Source: SpikeDetector/detector/evaluate_hybrid_janelia.py :: global_refractory_nms
def global_refractory_nms(
    indices: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    refractory_samples: int,
    timeline_length: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Apply score-priority NMS across chunk boundaries on a global timeline."""
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if indices.size != scores.size:
        raise ValueError("indices and scores must have equal length")
    if indices.size == 0:
        return indices, scores, 0
    valid = (indices >= 0) & (indices < int(timeline_length)) & np.isfinite(scores)
    indices, scores = indices[valid], scores[valid]
    if indices.size == 0:
        return indices, scores, 0

    # Exact duplicates can occur only at implementation boundaries; retain the
    # maximum score before applying the true refractory exclusion.
    sort_by_time = np.argsort(indices, kind="stable")
    indices, scores = indices[sort_by_time], scores[sort_by_time]
    unique_indices, first = np.unique(indices, return_index=True)
    max_scores = np.maximum.reduceat(scores, first)

    radius = max(0, int(refractory_samples))
    blocked = np.zeros(int(timeline_length), dtype=bool)
    selected: list[int] = []
    for candidate in np.argsort(-max_scores, kind="stable"):
        sample = int(unique_indices[candidate])
        if blocked[sample]:
            continue
        selected.append(int(candidate))
        blocked[max(0, sample - radius) : min(timeline_length, sample + radius + 1)] = True
    selected_arr = np.asarray(selected, dtype=np.int64)
    out_indices = unique_indices[selected_arr]
    out_scores = max_scores[selected_arr]
    chronological = np.argsort(out_indices, kind="stable")
    removed = int(indices.size - chronological.size)
    return out_indices[chronological], out_scores[chronological], removed

# Source: SpikeDetector/detector/evaluate_hybrid_janelia.py :: estimate_candidate_event_snrs_by_channel
def estimate_candidate_event_snrs_by_channel(
    bandpassed_signal: np.ndarray,
    indices: Sequence[int] | np.ndarray,
    noise_levels: Sequence[float] | np.ndarray,
    radius_samples: int,
    *,
    batch_size: int = 65536,
) -> np.ndarray:
    """Estimate deployable per-channel event SNRs without ground-truth labels.

    This mirrors the evaluator's unit-SNR convention, but operates on each
    candidate waveform independently: each channel's largest negative excursion
    is divided by that channel's robust recording noise. Windows are
    clipped at recording boundaries instead of padding with artificial values.
    """
    signal = np.asarray(bandpassed_signal, dtype=np.float32)
    event_indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    noise = np.asarray(noise_levels, dtype=np.float32).reshape(-1)
    radius = int(radius_samples)
    if signal.ndim != 2:
        raise ValueError(f"Expected signal [time, channel], got {signal.shape}")
    if noise.shape != (signal.shape[1],):
        raise ValueError(
            f"Expected one noise level per channel ({signal.shape[1]}), got {noise.shape}"
        )
    if radius < 0:
        raise ValueError("radius_samples must be non-negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not np.isfinite(signal).all():
        raise FloatingPointError("Bandpassed signal contains NaN/Inf")
    if not np.isfinite(noise).all() or bool((noise <= 0.0).any()):
        raise ValueError(f"Invalid robust noise levels: {noise}")
    if event_indices.size and (
        bool((event_indices < 0).any()) or bool((event_indices >= signal.shape[0]).any())
    ):
        raise ValueError("Candidate indices fall outside the signal timeline")

    output = np.empty((event_indices.size, signal.shape[1]), dtype=np.float32)
    offsets = np.arange(-radius, radius + 1, dtype=np.int64)
    for start in range(0, event_indices.size, int(batch_size)):
        stop = min(event_indices.size, start + int(batch_size))
        absolute = event_indices[start:stop, None] + offsets[None, :]
        valid = (absolute >= 0) & (absolute < signal.shape[0])
        clipped = np.clip(absolute, 0, signal.shape[0] - 1)
        windows = signal[clipped]
        windows = np.where(valid[..., None], windows, np.inf)
        negative_amplitude = np.maximum(0.0, -np.min(windows, axis=1))
        output[start:stop] = negative_amplitude / noise[None, :]
    return output

# Source: SpikeDetector/detector/evaluate_hybrid_janelia.py :: filter_scored_predictions_with_ranked_event_snr_gate
def filter_scored_predictions_with_ranked_event_snr_gate(
    indices: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    event_snrs_by_channel: np.ndarray,
    threshold: float,
    low_confidence_threshold: float,
    primary_snr_threshold: float,
    secondary_snr_threshold: float,
    min_secondary_channels: int = 1,
) -> tuple[np.ndarray, int]:
    """Gate low-confidence candidates by asymmetric ranked-channel SNRs.

    The primary channel is selected independently for every candidate as the
    channel with the largest local event SNR. A candidate passes the waveform
    check when that primary SNR reaches ``primary_snr_threshold`` and at least
    ``min_secondary_channels`` of the remaining channels reach
    ``secondary_snr_threshold``. High-confidence candidates bypass this check.
    """
    event_indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    event_scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    snrs = np.asarray(event_snrs_by_channel, dtype=np.float32)
    if snrs.ndim != 2:
        raise ValueError("event_snrs_by_channel must be [events, channels]")
    if event_indices.size != event_scores.size or event_indices.size != snrs.shape[0]:
        raise ValueError("indices, scores, and event SNRs must have equal event counts")
    if snrs.shape[1] < 2:
        raise ValueError("Ranked event-SNR gating requires at least two channels")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if not 0.0 <= float(low_confidence_threshold) <= 1.0:
        raise ValueError("low_confidence_threshold must be in [0, 1]")
    if float(primary_snr_threshold) < 0.0 or float(secondary_snr_threshold) < 0.0:
        raise ValueError("SNR thresholds must be non-negative")
    secondary_count = int(min_secondary_channels)
    if not 1 <= secondary_count < snrs.shape[1]:
        raise ValueError(
            f"min_secondary_channels must be in [1, {snrs.shape[1] - 1}], "
            f"got {secondary_count}"
        )
    if not np.isfinite(snrs).all():
        raise FloatingPointError("event_snrs_by_channel contains NaN/Inf")

    # Descending rank 0 is the dynamic primary channel. Rank k is the kth
    # strongest distinct secondary channel, so no fixed channel identity leaks
    # into the decision rule.
    ranked = np.sort(snrs, axis=1)[:, ::-1]
    waveform_pass = (
        (ranked[:, 0] >= float(primary_snr_threshold))
        & (ranked[:, secondary_count] >= float(secondary_snr_threshold))
    )
    eligible = event_scores >= float(threshold)
    double_low = (event_scores < float(low_confidence_threshold)) & ~waveform_pass
    keep = eligible & ~double_low
    rejected = int(np.count_nonzero(eligible & double_low))
    return np.ascontiguousarray(event_indices[keep], dtype=np.int64), rejected

# Source: SpikeDetector/detector/evaluate_hybrid_janelia.py :: detect_spikes_from_prob_vectorized
def detect_spikes_from_prob_vectorized(
    prob: np.ndarray, threshold: float, peak_radius: int, refractory_samples: int
) -> np.ndarray:
    """Equivalent local maxima and score-ordered NMS with a vectorized peak scan."""
    if prob.size == 0:
        return np.empty(0, dtype=np.int64)
    if peak_radius < 0 or not np.isfinite(prob).all():
        return detect_spikes_from_prob(prob, threshold, peak_radius, refractory_samples)
    local_max = maximum_filter1d(
        prob, size=2 * peak_radius + 1, mode="constant", cval=-np.inf
    )
    peaks = np.flatnonzero((prob >= threshold) & (prob >= local_max))
    order = peaks[np.argsort(prob[peaks])[::-1]]
    blocked = np.zeros(prob.size, dtype=bool)
    selected = []
    for t in order:
        if blocked[t]:
            continue
        selected.append(t)
        left = max(0, t - refractory_samples)
        right = min(prob.size, t + refractory_samples + 1)
        blocked[left:right] = True
    return np.asarray(sorted(selected), dtype=np.int64)

# Source: SpikeDetector/detector/detector_utils.py :: detect_spikes_from_prob
def detect_spikes_from_prob(prob: np.ndarray,
                            threshold: float,
                            peak_radius: int,
                            refractory_samples: int) -> np.ndarray:
    """
    prob: [T] float in [0,1]
    返回 spike_times (sample indices)
    """
    T = prob.shape[0]
    cand = np.where(prob >= threshold)[0]
    if cand.size == 0:
        return np.array([], dtype=np.int64)

    is_peak = np.zeros(T, dtype=bool)
    for t in cand:
        left = max(0, t - peak_radius)
        right = min(T, t + peak_radius + 1)
        if prob[t] >= prob[left:right].max():
            is_peak[t] = True

    peaks = np.where(is_peak)[0]
    if peaks.size == 0:
        return np.array([], dtype=np.int64)

    order = peaks[np.argsort(prob[peaks])[::-1]]
    blocked = np.zeros(T, dtype=bool)
    selected = []
    for t in order:
        if blocked[t]:
            continue
        selected.append(t)
        left = max(0, t - refractory_samples)
        right = min(T, t + refractory_samples + 1)
        blocked[left:right] = True

    return np.array(sorted(selected), dtype=np.int64)

# Source: Spike/utils/spikeforest_utils.py :: interpolate_spike_pytorch
def interpolate_spike_pytorch(data, target_length=121, mode='linear'):
    """
    data: Tensor of shape [batch_size, 60, channels]
    Returns: Tensor of shape [batch_size, 121, channels]
    """
    # PyTorch 的 interpolate 要求 shape 为 [B, C, T]
    data = data.permute(0, 2, 1)  # [B, C, 60]

    out = F.interpolate(data, size=target_length, mode=mode, align_corners=False)

    return out.permute(0, 2, 1)  # [B, 121, C]

# Source: Spike/utils/spikeforest_utils.py :: repeat_channels
def repeat_channels(x: torch.Tensor, repeat_factor: int, max_channels=11) -> torch.Tensor:
    """
    Repeat each channel 'repeat_factor' times along the channel dimension.

    Args:
        x: Tensor of shape [bs, time, channels]
        repeat_factor: How many times to repeat each channel

    Returns:
        Tensor of shape [bs, time, channels * repeat_factor]
    """
    bs, time, ch = x.shape
    # min_channel = get_channels_with_min_value(x)
    # repeat_times = max(0, repeat_factor * ch - max_channels)

    # Step 1: reshape to [bs, time, ch, 1]
    x = x.unsqueeze(-1)  # [bs, time, ch, 1]

    # Step 2: repeat last dimension
    x = x.repeat(1, 1, 1, repeat_factor)  # [bs, time, ch, repeat_factor]

    # Step 3: reshape back to [bs, time, ch * repeat_factor]
    x = x.reshape(bs, time, ch * repeat_factor)
    return x

# Source: Spike/utils/spikeforest_utils.py :: preprocess_spikeforest_data
def preprocess_spikeforest_data(data,
                                normalize=True,
                                interpolate=90,
                                repeat=True,
                                repeat_times=3,
                                max_channels=11,
                                pad_len=None):
    data = torch.tensor(data, dtype=torch.float).detach()
    if normalize:
        data = _zscore_normalize(data)
    if interpolate:
        data = interpolate_spike_pytorch(data, mode='linear', target_length=interpolate)
    if repeat:
        if pad_len is not None:
            new_data = []
            for i in range(data.shape[0]):
                temp_data = data[i, :, :pad_len[i]].unsqueeze(0)
                # print(temp_data.shape)
                if temp_data.shape[-1] < max_channels:
                    repeat_times = (max_channels // temp_data.shape[2]) + 1
                    temp_data = repeat_channels(temp_data, repeat_times)
                # else:
                #     center = temp_data.shape[2] // 2
                #     left = center - 6 // 2
                #     right = left + 6
                #     temp_data = temp_data[:, :, left:right]
                #     temp_data = repeat_channels(temp_data, 2)
                center = temp_data.shape[2] // 2
                left = center - max_channels // 2
                right = left + max_channels
                temp_data = temp_data[:, :, left:right]
                new_data.append(temp_data)
            new_data = torch.cat(new_data, dim=0)
            return new_data

        if data.shape[2] < max_channels:
            repeat_times  = (max_channels // data.shape[2]) + 1
            residual = data.shape[2] * repeat_times - max_channels
            if residual > repeat_times + 1:
                pad_length = max_channels - data.shape[2] * (repeat_times - 1)
                data = repeat_channels(data, repeat_times - 1)
                data = data.permute(0, 2, 1)
                data = F.pad(data, pad=(0, 0, pad_length // 2, pad_length - (pad_length // 2)), mode='replicate')
                data = data.permute(0, 2, 1)
            else:
                data = repeat_channels(data, repeat_times)
            # data = repeat_channels(data, repeat_times)
        center = data.shape[2] // 2
        left = center - max_channels // 2
        right = left + max_channels
        data = data[:, :, left:right,]
    return data

# Source: Spike/utils/spikeforest_utils.py :: _zscore_normalize
def _zscore_normalize(data, dim=(0, 1)):
    mean = torch.mean(data, dim=dim, keepdim=True)
    std = torch.std(data, dim=dim, keepdim=True)
    data = (data - mean) / (std + 1e-6)

    return data * 1.0

# Source: Spike/experiments/temporal_geometry_gmm.py :: amplitude_contrasts
def amplitude_contrasts(ptp: np.ndarray) -> tuple[np.ndarray, dict]:
    """Scale-invariant PTP fractions projected onto a C-1 Helmert basis.

    No raw amplitude/log-amplitude/latency is added in this single-variable
    ablation. Zero-amplitude rows become uniform fractions, not NaNs. The
    standardizer is fitted on this recording without labels, just as for the
    frozen HuiduRep embedding; its transform is not refitted per temporal bin.
    """
    a = np.asarray(ptp, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] < 2 or not len(a):
        raise ValueError('Need nonempty [events,physical_channels] amplitudes')
    if not np.isfinite(a).all() or np.any(a < 0):
        raise ValueError('PTP must be finite and nonnegative')
    total = a.sum(axis=1, keepdims=True)
    fractions = np.divide(a, total, out=np.full_like(a, 1 / a.shape[1]), where=total > 0)
    basis = helmert(a.shape[1], full=False)
    contrasts = fractions @ basis.T
    scaler = StandardScaler().fit(contrasts)
    return np.ascontiguousarray(scaler.transform(contrasts)), {
        'physical_channels': a.shape[1], 'features_added': a.shape[1] - 1,
        'fraction_mean': fractions.mean(axis=0).tolist(),
        'basis': basis.tolist(), 'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist(), 'zero_amplitude_rows': int((total == 0).sum()),
        'definition': 'PTP/sum(PTP), Helmert contrasts, per-record StandardScaler',
    }

# Source: Spike/experiments/template_residual_gmm.py :: ResidualConfig
@dataclass(frozen=True)
class ResidualConfig:
    """Fixed before scoring rec11; no GT-based parameter search in this ablation."""

    before: int = 30
    after: int = 30
    max_shift: int = 2
    amplitude_min: float = 0.5
    amplitude_max: float = 2.0
    core_posterior: float = 0.9
    min_core_events: int = 30
    max_template_events: int = 512
    fold_block_seconds: float = 1.0
    top_components: int = 3
    residual_weight: float = 2.0
    max_log_evidence: float = 4.0
    batch_size: int = 512
    null_permutation_seed: int = 9090

    def validate(self) -> None:
        integer_fields = ('before', 'after', 'max_shift', 'min_core_events',
                          'max_template_events', 'top_components', 'batch_size',
                          'null_permutation_seed')
        for key in integer_fields:
            value = getattr(self, key)
            if not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f'{key} must be a nonnegative integer')
        if min(self.before, self.after, self.min_core_events, self.top_components,
               self.batch_size) < 1 or self.max_template_events < self.min_core_events:
            raise ValueError('Invalid window, core event count, or batch size')
        scalars = (self.amplitude_min, self.amplitude_max, self.core_posterior,
                   self.fold_block_seconds, self.residual_weight, self.max_log_evidence)
        if not np.isfinite(scalars).all():
            raise ValueError('Nonfinite residual configuration')
        if not 0 < self.amplitude_min <= self.amplitude_max:
            raise ValueError('Amplitude bounds must be positive and ordered')
        if not 0 < self.core_posterior <= 1 or self.fold_block_seconds <= 0:
            raise ValueError('Invalid posterior or fold duration')
        if self.residual_weight < 0 or self.max_log_evidence < 0:
            raise ValueError('Residual strength must be nonnegative')

# Source: Spike/experiments/template_residual_gmm.py :: extract_padded_waveforms
def extract_padded_waveforms(
    traces: np.ndarray, samples: np.ndarray, config: ResidualConfig,
) -> np.ndarray:
    """Gather [N, before+after+2*shift, C]; strict bounds, no padded fake data."""
    config.validate()
    traces, samples = np.asarray(traces), np.asarray(samples)
    if traces.ndim != 2 or traces.shape[1] < 1 or samples.ndim != 1:
        raise ValueError('Expected traces[T,C] and samples[N]')
    if samples.dtype.kind not in 'iu' or len(samples) == 0:
        raise ValueError('Need nonempty integer sample indices')
    offsets = np.arange(-config.before-config.max_shift, config.after+config.max_shift)
    if np.min(samples)+offsets[0] < 0 or np.max(samples)+offsets[-1] >= len(traces):
        raise ValueError('Waveform/shift window outside recording')
    out = np.empty((len(samples), len(offsets), traces.shape[1]), dtype=np.float32)
    for start in range(0, len(samples), config.batch_size):
        ids = samples[start:start+config.batch_size, None]+offsets[None]
        out[start:start+len(ids)] = traces[ids]
    if not np.isfinite(out).all():
        raise ValueError('Nonfinite waveform')
    return out

# Source: Spike/experiments/template_residual_gmm.py :: block_folds
def block_folds(samples: np.ndarray, fs_hz: float, seconds: float) -> np.ndarray:
    """Alternate whole one-second blocks, avoiding interleaved overlapping snippets."""
    samples = np.asarray(samples)
    if samples.ndim != 1 or samples.dtype.kind not in 'iu' or np.any(samples < 0):
        raise ValueError('Invalid event sample clock')
    if not np.isfinite([fs_hz, seconds]).all() or fs_hz <= 0 or seconds <= 0:
        raise ValueError('Invalid sampling rate/fold duration')
    block = int(round(fs_hz*seconds))
    if block < 1:
        raise ValueError('Fold duration below one sample')
    return ((samples//block) % 2).astype(np.int8)

# Source: Spike/experiments/template_residual_gmm.py :: frozen_log_joint
def frozen_log_joint(x: np.ndarray, parameters: dict[str, np.ndarray]) -> np.ndarray:
    """Full Gaussian log probabilities from saved A states; no sklearn private API."""
    x = np.asarray(x, dtype=np.float64)
    means, covs, weights = (np.asarray(parameters[key], dtype=np.float64)
                            for key in ('means', 'covariances', 'weights'))
    if x.ndim != 2 or means.shape != (len(weights), x.shape[1]):
        raise ValueError('Feature/mean dimensions differ')
    if covs.shape != (len(weights), x.shape[1], x.shape[1]):
        raise ValueError('Covariance dimensions differ')
    if not all(np.isfinite(a).all() for a in (x, means, covs, weights)):
        raise ValueError('Nonfinite GMM input')
    if np.any(weights <= 0) or not np.isclose(weights.sum(), 1.):
        raise ValueError('Invalid mixture weights')
    out = np.empty((len(x), len(weights)), dtype=np.float64)
    normalizer = x.shape[1]*np.log(2*np.pi)
    for k in range(len(weights)):
        factor = cholesky(covs[k], lower=True)
        whitened = solve_triangular(factor, (x-means[k]).T, lower=True)
        out[:, k] = (np.log(weights[k])-.5*(normalizer+
            2*np.log(factor.diagonal()).sum()+np.square(whitened).sum(axis=0)))
    return out

# Source: Spike/experiments/template_residual_gmm.py :: crossfit_templates
def crossfit_templates(
    waveforms: np.ndarray, labels: np.ndarray, core_confidence: np.ndarray,
    folds: np.ndarray, k: int, config: ResidualConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median templates [target fold,K,L,C], from the *opposite* fold only.

    No event-wise channel standardization. Subtract a DC mean from each physical
    channel, retaining amplitude ratios. Subsampling cores is time-spread and
    deterministic (input events must be sorted by sample).
    """
    config.validate()
    n, width, channels = waveforms.shape
    length = config.before+config.after
    if width != length+2*config.max_shift:
        raise ValueError('Wrong padded waveform width')
    if any(np.shape(v) != (n,) for v in (labels, core_confidence, folds)):
        raise ValueError('Core metadata length mismatch')
    if (np.any(labels < 0) or np.any(labels >= k) or
            not np.isin(folds, [0, 1]).all() or
            not np.isfinite(core_confidence).all() or
            np.any(core_confidence < 0) or np.any(core_confidence > 1)):
        raise ValueError('Invalid labels/folds/posteriors')
    templates = np.zeros((2, k, length, channels), dtype=np.float64)
    counts = np.zeros((2, k), dtype=np.int64)
    used = np.zeros_like(counts)
    for target_fold in (0, 1):
        for unit in range(k):
            ids = np.flatnonzero((folds != target_fold) & (labels == unit) &
                                 (core_confidence >= config.core_posterior))
            counts[target_fold, unit] = len(ids)
            if len(ids) < config.min_core_events:
                continue
            if len(ids) > config.max_template_events:
                ids = ids[np.linspace(0, len(ids)-1, config.max_template_events).astype(int)]
            w = np.asarray(waveforms[ids, config.max_shift:config.max_shift+length], dtype=np.float64)
            if not np.isfinite(w).all():
                raise ValueError('Nonfinite core waveform')
            w -= w.mean(axis=1, keepdims=True)
            template = np.median(w, axis=0)
            template -= template.mean(axis=0, keepdims=True)
            templates[target_fold, unit] = template
            used[target_fold, unit] = len(ids)
    return templates, counts, used

# Source: Spike/experiments/template_residual_gmm.py :: minimum_residual
def minimum_residual(
    padded: np.ndarray, templates: np.ndarray, config: ResidualConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Best mean-square residual [B,J], bounded shared amplitude and integer shift.

    `templates` is [B,J,L,C]; one shared positive scale per event/template keeps
    physical channel relationships. Shifts are window crops, never np.roll.
    The returned shift is diagnostic ONLY: event timestamps remain unchanged.
    """
    length = config.before+config.after
    if padded.ndim != 3 or templates.ndim != 4 or templates.shape[0] != len(padded):
        raise ValueError('Expected padded[B,W,C], templates[B,J,L,C]')
    if padded.shape[1] != length+2*config.max_shift or templates.shape[2:] != (length, padded.shape[2]):
        raise ValueError('Residual waveform/template window mismatch')
    norm = np.square(templates).sum(axis=(2, 3))
    error = np.full(norm.shape, np.inf)
    amplitude = np.ones_like(error)
    best_shift = np.zeros(error.shape, dtype=np.int8)
    # Prefer zero shift when fits tie exactly, then smaller displacement.
    shifts = [0]+[d for radius in range(1, config.max_shift+1) for d in (-radius, radius)]
    for shift in shifts:
        start = config.max_shift+shift
        w = np.asarray(padded[:, start:start+length], dtype=np.float64).copy()
        w -= w.mean(axis=1, keepdims=True)
        dot = np.einsum('blc,bjlc->bj', w, templates, optimize=False)
        scale = np.clip(dot/np.maximum(norm, 1e-20), config.amplitude_min, config.amplitude_max)
        e = (np.square(w).sum(axis=(1, 2))[:, None]-2*scale*dot+scale**2*norm)/(length*padded.shape[2])
        e = np.maximum(e, 0.)  # floating-point cancellation at an exact fit
        improve = e < error
        error[improve] = e[improve]
        amplitude[improve] = scale[improve]
        best_shift[improve] = shift
    if not np.isfinite(error).all():
        raise ValueError('Nonfinite residual')
    return error, amplitude, best_shift

# Source: Spike/experiments/template_residual_gmm.py :: residual_assign
def residual_assign(
    log_joint: np.ndarray, waveforms: np.ndarray, folds: np.ndarray,
    templates: np.ndarray, counts: np.ndarray, config: ResidualConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Single bounded reassignment pass; K/count/timestamps never change.

    Compare top J feature components. Score = log p_A(k,z) +
    clip(weight*(E_at_A-E_k), -cap, cap). Missing/degenerate template for any
    candidate => original A for that event, not a penalty for an unseen unit.
    """
    config.validate()
    n, k = log_joint.shape
    if len(waveforms) != n or np.shape(folds) != (n,) or templates.shape[:2] != (2, k):
        raise ValueError('Assignment input shape mismatch')
    if np.shape(counts) != (2, k) or not np.isfinite(log_joint).all():
        raise ValueError('Invalid template counts or log probabilities')
    labels_a = np.argmax(log_joint, axis=1).astype(np.int32)
    labels = labels_a.copy()
    eligible_total = 0
    changed_error_gain: list[np.ndarray] = []
    changed_prior_cost: list[np.ndarray] = []
    shifted = 0
    amplitude_at_bound = 0
    j = min(config.top_components, k)
    valid = ((counts >= config.min_core_events) &
             (np.square(templates).sum(axis=(2, 3)) > 1e-12))
    if config.residual_weight > 0 and config.max_log_evidence > 0:
        for begin in range(0, n, config.batch_size):
            end = min(n, begin+config.batch_size)
            joint = log_joint[begin:end]
            top = np.argsort(-joint, axis=1, kind='stable')[:, :j]
            f = folds[begin:end]
            eligible = np.all(valid[f[:, None], top], axis=1)
            if not np.any(eligible):
                continue
            rows = np.flatnonzero(eligible)
            top = top[rows]
            eligible_total += len(rows)
            errors, scales, shifts = minimum_residual(
                waveforms[begin+rows], templates[f[rows, None], top], config)
            feature_score = np.take_along_axis(joint[rows], top, axis=1)
            bonus = np.clip(config.residual_weight*(errors[:, :1]-errors),
                            -config.max_log_evidence, config.max_log_evidence)
            selected = np.argmax(feature_score+bonus, axis=1)
            winner = top[np.arange(len(rows)), selected]
            labels[begin+rows] = winner
            changed = selected != 0
            rr = np.flatnonzero(changed)
            if len(rr):
                chosen = selected[rr]
                changed_error_gain.append(errors[rr, 0]-errors[rr, chosen])
                changed_prior_cost.append(feature_score[rr, 0]-feature_score[rr, chosen])
                shifted += int(np.count_nonzero(shifts[rr, chosen]))
                amp = scales[rr, chosen]
                amplitude_at_bound += int(np.count_nonzero(
                    np.isclose(amp, config.amplitude_min) | np.isclose(amp, config.amplitude_max)))
    changed_count = int(np.count_nonzero(labels != labels_a))
    def describe(values):
        return np.quantile(np.concatenate(values), [0, .25, .5, .75, 1]).tolist() if values else None
    audit = dict(events=n,changed_events=changed_count,changed_fraction=changed_count/n,
        eligible_events=eligible_total,invalid_template_slots=int(np.count_nonzero(~valid)),
        changed_residual_gain_quantiles=describe(changed_error_gain),
        changed_log_prior_cost_quantiles=describe(changed_prior_cost),
        changed_events_nonzero_shift=shifted,changed_events_amplitude_at_bound=amplitude_at_bound,
        noise_events=0,assigned_units=int(np.unique(labels).size),
        event_times_modified=False,iterations=1,procedure_completed=True)
    return labels, audit

# Source: Spike/experiments/weighted_template_d.py :: _lowest_score_mask
def _lowest_score_mask(scores: np.ndarray, count: int, tie_seed: int) -> np.ndarray:
    """Exact rank count with seeded ties; identical to previous hard-exclusion selection."""
    selected = np.zeros(len(scores), dtype=bool)
    if not 0 <= count <= len(scores):
        raise ValueError("Invalid low-rank count")
    if not count:
        return selected
    boundary = np.partition(scores, count-1)[count-1]
    selected[scores < boundary] = True
    remaining = count-int(selected.sum())
    tied_ids = np.flatnonzero(scores == boundary)
    chosen = (tied_ids if remaining == len(tied_ids) else
              np.random.default_rng(tie_seed).choice(tied_ids, remaining, replace=False))
    selected[chosen] = True
    return selected
