"""Standalone frozen detector -> HuiduRep -> VanillaCluster inference.

Inputs are local arrays, not historical score/embedding caches. No ground truth,
network access, original repository, training dataset or NAS is required.
"""

from __future__ import annotations

import gc
import warnings

import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch

from . import ops
from .checkpoints import load_detector, load_huidurep


def validate_input(raw, coords, fs_hz):
    if raw.ndim != 2 or raw.shape[1] != 4 or len(raw) < 100:
        raise ValueError("Expected at least 100 raw samples with shape [time,4]")
    if not np.isfinite(raw).all() or np.shape(coords) != (4, 2) or not np.isfinite(coords).all():
        raise ValueError("Input/coordinates must be finite; coords shape is [4,2]")
    if not np.isclose(fs_hz, 30000.0, rtol=0, atol=1.0):
        raise ValueError("This frozen model expects approximately 30 kHz; no silent resampling is performed")


def filtered_signal(raw, fs, config):
    # Exact operation order and float32 cast points of the original D1 runner.
    sos = butter(config["order"], (config["low"], config["high"]), btype="bandpass", fs=fs, output="sos")
    result = sosfiltfilt(sos, raw, axis=0).astype(np.float32)
    if config["notch"]:
        b, a = iirnotch(60.0, 30.0, fs=fs)
        result = filtfilt(b, a, result, axis=0).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite filter output")
    return result


def normalize(signal):
    location, scale = ops.robust_location_scale_time_channel(signal)
    return ops.robust_normalize_time_channel(signal, location=location, scale=scale), {
        "median": location.ravel().tolist(),
        "robust_scale": scale.ravel().tolist(),
    }


@torch.inference_mode()
def infer_candidates(model, signal, coords, config, device, batch_size=8):
    """Original chunk-local peaks/NMS then global NMS, before threshold/SNR gate."""
    start, chunk = 0, config["chunk_len"]
    times, scores, retries = [], [], []
    initial_batch = batch_size
    while start < len(signal):
        remaining = len(signal) - start
        count = min(batch_size, remaining // chunk)
        length = chunk if count else remaining
        count = max(count, 1)
        end = start + count * length
        x = c = logits = probability = None
        try:
            x = torch.from_numpy(signal[start:end].reshape(count, length, 4)).to(device)
            c = torch.from_numpy(coords).unsqueeze(0).expand(count, -1, -1).to(device)
            logits = model(x, c)
            if logits.shape != (count, length) or not torch.isfinite(logits).all():
                raise ValueError("Invalid detector output")
            probability = logits.float().sigmoid().cpu().numpy()
        except torch.cuda.OutOfMemoryError:
            retries.append({"start": start, "batch_size": batch_size})
            del x, c, logits, probability
            gc.collect()
            torch.cuda.empty_cache()
            if count <= 1:
                raise RuntimeError("One detector chunk does not fit: retry with --device cpu") from None
            batch_size = max(1, count // 2)
            continue
        for j, p in enumerate(probability):
            local = ops.detect_spikes_from_prob_vectorized(
                p, config["candidate_floor"], config["peak_radius"], config["refractory_samples"]
            )
            times.append(local + start + j * length)
            scores.append(p[local])
        start = end
        del x, c, logits, probability
    times, scores, removed = ops.global_refractory_nms(
        np.concatenate(times), np.concatenate(scores), config["refractory_samples"], len(signal)
    )
    return (
        times,
        scores,
        {
            "initial_batch_size": initial_batch,
            "final_batch_size": batch_size,
            "oom_retries": retries,
            "boundary_removed": removed,
            "tail_samples": len(signal) % chunk,
        },
    )


@torch.inference_mode()
def embeddings(model, inputs, device, batch_size=128):
    start, output, retries = 0, [], []
    decoder_calls = [0]

    def hook(module, args, value):
        decoder_calls[0] += 1

    handle = model.feature_decoder.register_forward_hook(hook)
    successful = 0
    try:
        while start < len(inputs):
            x = values = None
            count = min(batch_size, len(inputs) - start)
            before = decoder_calls[0]
            try:
                x = inputs[start : start + count].to(device)
                values = model.transform(x, None)
                if decoder_calls[0] - before != 1:
                    raise ValueError("Expected one internal denoise per HuiduRep transform")
                if values.shape != (count, 32) or not torch.isfinite(values).all():
                    raise ValueError("Invalid HuiduRep output")
                output.append(values.cpu().numpy())
            except torch.cuda.OutOfMemoryError:
                retries.append({"start": start, "batch_size": batch_size})
                del x, values
                gc.collect()
                torch.cuda.empty_cache()
                if count <= 1:
                    raise RuntimeError("One embedding does not fit: retry with --device cpu") from None
                batch_size = max(1, count // 2)
                continue
            start += count
            successful += 1
            del x, values
    finally:
        handle.remove()
    raw = np.concatenate(output)
    scaler = StandardScaler().fit(raw)
    features = scaler.transform(raw).astype(np.float32)
    return features, {
        "successful_batches": successful,
        "decoder_calls": decoder_calls[0],
        "final_batch_size": batch_size,
        "oom_retries": retries,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }


def strict_trim_templates(waves, labels, posterior, folds, scores, k, config, fraction=0.25, tie_seed=20260912):
    """Binary trim25 specialization of the historical weighted-template path."""
    templates, counts, used = ops.crossfit_templates(waves, labels, posterior, folds, k, config)
    details = []
    for fold in (0, 1):
        for unit in range(k):
            ids = np.flatnonzero((folds != fold) & (labels == unit) & (posterior >= config.core_posterior))
            if len(ids) < config.min_core_events:
                continue
            if len(ids) > config.max_template_events:
                ids = ids[np.linspace(0, len(ids) - 1, config.max_template_events).astype(int)]
            dropped = ops._lowest_score_mask(
                scores[ids], int(np.floor(fraction * len(ids))), tie_seed + 2 * unit + fold
            )
            retained = ids[~dropped]
            counts[fold, unit] = len(retained)
            fallback = len(retained) < config.min_core_events
            if not fallback:
                w = np.asarray(
                    waves[retained, config.max_shift : config.max_shift + config.before + config.after],
                    dtype=np.float64,
                ).copy()
                w -= w.mean(axis=1, keepdims=True)
                template = np.median(w, axis=0)
                template -= template.mean(axis=0, keepdims=True)
                fallback = np.square(template).sum() <= 1e-12
                if not fallback:
                    templates[fold, unit] = template
            if fallback:
                templates[fold, unit] = 0
                counts[fold, unit] = 0
            details.append(
                {
                    "target_fold": fold,
                    "cluster": unit,
                    "core_pool": len(ids),
                    "retained": len(retained),
                    "fallback_C": bool(fallback),
                }
            )
    return templates, counts, details


def cluster(features, auxiliary, waves, ids, samples, scores, fs, components, seed, config):
    if len(features) < components:
        raise ValueError(f"{len(features)} candidates are fewer than K={components}; do not silently change K")
    x = np.ascontiguousarray(np.column_stack((features.astype(np.float64), auxiliary)))
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gmm = GaussianMixture(n_components=components, random_state=seed, **config["gmm"]).fit(x)
        initial = gmm.predict(x).astype(np.int32)
        final = initial.copy()
        parameters = dict(means=gmm.means_, covariances=gmm.covariances_, weights=gmm.weights_)
        residual_config = ops.ResidualConfig(**config["residual"])
        audit = {"eligible_events": 0, "changed_events": 0}
        templates = np.zeros((2, components, 60, 4), dtype=np.float64)
        counts = np.zeros((2, components), dtype=np.int64)
        details = []
        if len(ids):
            joint = ops.frozen_log_joint(x[ids], parameters)
            np.testing.assert_array_equal(joint.argmax(axis=1), initial[ids])
            posterior = np.exp(np.max(joint - logsumexp(joint, axis=1, keepdims=True), axis=1))
            folds = ops.block_folds(samples[ids], fs, residual_config.fold_block_seconds)
            templates, counts, details = strict_trim_templates(
                waves,
                initial[ids],
                posterior,
                folds,
                scores[ids],
                components,
                residual_config,
                config["template_low_fraction"],
                config["template_tie_seed"],
            )
            final[ids], audit = ops.residual_assign(joint, waves, folds, templates, counts, residual_config)
        audit.update(
            seed=seed,
            components=components,
            converged=bool(gmm.converged_),
            iterations=int(gmm.n_iter_),
            gmm_lower_bound=float(gmm.lower_bound_),
            warnings=[str(w.message) for w in caught],
            template_audits=details,
            boundary_fallback_D_to_C=len(samples) - len(ids),
        )
    return (
        dict(
            labels=final,
            initial_labels=initial,
            templates=templates,
            core_counts=counts,
            means=parameters["means"],
            covariances=parameters["covariances"],
            weights=parameters["weights"],
        ),
        audit,
    )


def run_pipeline(
    raw,
    coords,
    fs,
    config,
    device,
    components,
    seed=30,
    profile="d1",
    detector_batch=8,
    embedding_batch=128,
    detect_only=False,
    verbose=True,
    owner_channels=None,
):
    validate_input(raw, coords, fs)
    raw = np.asarray(raw, dtype=np.float32)
    coords = np.asarray(coords, dtype=np.float32)
    coords = np.ascontiguousarray(coords - coords.mean(axis=0, keepdims=True))
    p = config["profiles"][profile]
    if verbose:
        print(f"Preprocessing {len(raw)/fs:.3f}s, profile={profile}", flush=True)
    canonical, gate_norm = normalize(filtered_signal(raw, fs, config["profiles"]["canonical"]["filter"]))
    if profile == "canonical":
        physical = canonical
        detector_signal, detector_norm = canonical, gate_norm
    else:
        physical = filtered_signal(raw, fs, p["filter"])
        detector_signal, detector_norm = normalize(physical)
    model = load_detector(config, device)
    times, values, det_audit = infer_candidates(model, detector_signal, coords, config, device, detector_batch)
    del model, detector_signal
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    snrs = ops.estimate_candidate_event_snrs_by_channel(
        canonical, times, np.ones(4, np.float32), config["event_snr_radius"]
    )
    pred, rejected = ops.filter_scored_predictions_with_ranked_event_snr_gate(
        times,
        values,
        snrs,
        p["base_threshold"],
        p["direct_keep"],
        config["primary_snr"],
        config["secondary_snr"],
        config["min_secondary_channels"],
    )
    positions = np.searchsorted(times, pred)
    # For overlapping probe neighborhoods, each physical channel owns one patch.
    # The four-channel path keeps every accepted detection, exactly as published.
    ownership_removed = 0
    if owner_channels is not None and len(positions):
        owned = np.isin(np.argmax(snrs[positions], axis=1), owner_channels)
        ownership_removed = int((~owned).sum())
        pred, positions = pred[owned], positions[owned]
    scores = values[positions]
    selected_snrs = snrs[positions]
    arrays = dict(detection_samples=pred, detection_scores=scores, detection_snrs=selected_snrs)
    audit = dict(
        profile=profile,
        frames=len(raw),
        fs_hz=fs,
        candidate_peaks=len(times),
        detector_events=len(pred),
        gate_rejected=rejected,
        detector=det_audit,
        detector_normalization=detector_norm,
        gate_normalization=gate_norm,
        ownership_removed=ownership_removed,
        ground_truth_used=False,
    )
    if verbose:
        print(f"Detector: {len(pred)} accepted events", flush=True)
    del canonical, times, values, snrs
    if detect_only:
        return arrays, audit
    keep = (pred >= 30) & (pred < len(raw) - 30)
    samples, scores = pred[keep], scores[keep]
    if not len(samples):
        arrays.update(
            samples=samples, scores=scores, labels=np.empty(0, np.int32), initial_labels=np.empty(0, np.int32)
        )
        audit.update(sorting_events=0, clustering_status="no_complete_waveform_candidates")
        return arrays, audit
    waves = physical[samples[:, None] + np.arange(-30, 30)]
    # Preserve physical channel identities for the BaseSorting adapter.
    amplitudes = np.ptp(waves, axis=1)
    auxiliary, amp_audit = ops.amplitude_contrasts(np.ptp(waves, axis=1).astype(np.float64))
    residual_config = ops.ResidualConfig(**config["residual"])
    ids = np.flatnonzero((samples >= 32) & (samples + 32 <= len(raw)))
    padded = (
        ops.extract_padded_waveforms(physical, samples[ids], residual_config)
        if len(ids)
        else np.empty((0, 64, 4), np.float32)
    )
    del physical
    inputs = ops.preprocess_spikeforest_data(
        waves, normalize=True, interpolate=90, repeat=True, repeat_times=3, max_channels=11
    )
    del waves
    model = load_huidurep(config, device)
    if verbose:
        print(f"HuiduRep: {len(samples)} waveforms; denoise ON", flush=True)
    features, embed_audit = embeddings(model, inputs, device, embedding_batch)
    del model, inputs
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if verbose:
        print(f"VanillaCluster: K={components}, seed={seed}", flush=True)
    result, cluster_audit = cluster(features, auxiliary, padded, ids, samples, scores, fs, components, seed, config)
    arrays.update(samples=samples, scores=scores, features=features, auxiliary=auxiliary, **result)
    arrays["main_channel_indices"] = np.array(
        [
            (
                np.argmax(np.median(amplitudes[result["labels"] == unit], axis=0))
                if np.any(result["labels"] == unit)
                else -1
            )
            for unit in range(components)
        ],
        dtype=np.int64,
    )
    audit.update(
        sorting_events=len(samples),
        boundary_excluded=int((~keep).sum()),
        embedding=embed_audit,
        amplitude_features=amp_audit,
        clustering=cluster_audit,
        waveform_source=(
            "D1 filtered physical signal without global MAD"
            if profile == "d1"
            else "canonical filtered signal with global MAD"
        ),
    )
    return arrays, audit
