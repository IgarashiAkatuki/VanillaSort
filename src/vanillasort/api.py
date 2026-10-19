"""SpikeInterface recording/sorting adapter for the canonical inference pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spikeinterface.core import BaseRecording, BaseSorting


def sort(
    recording: BaseRecording,
    output_folder=None,
    model="default",
    device="auto",
    seed=0,
    verbose=True,
    *,
    model_path=None,
    components=22,
    profile="d1",
    enable_gate=True,
    threshold=None,
    gate_fn=None,
    direct_keep=None,
    primary_snr=None,
    secondary_snr=None,
    min_secondary_channels=None,
    detector_batch=8,
    embedding_batch=128,
    neighborhood_mode="dense",
    target_units=None,
    postprocessing=None,
) -> BaseSorting:
    """Sort a raw 30 kHz recording and return a SpikeInterface BaseSorting.

    Parameters
    ----------
    recording : spikeinterface.core.BaseRecording
        Raw traces on a common voltage scale, with finite 2D channel locations
        in micrometres. Each group/shank must contain at least four channels.
        Larger groups use experimental nearest-four neighborhoods.
    output_folder : str or Path, optional
        New/empty folder for sorting/, events.npz, per-patch results and run.json.
        If omitted, the returned sorting is held in memory; no files are written.
    model : str, default "default"
        Hybrid Janelia model, downloaded from a pinned Hugging Face revision on
        first use and cached locally. Ordinary package import does not download.
    device : str, default "auto"
        "cpu", "cuda", "cuda:N", or "auto" (CUDA if available, otherwise CPU).
    seed : int, default 0
        Fixed GMM initialization and PyTorch seed. No automatic K selection.
    verbose : bool, default True
        Print progress for preprocessing, detection, embedding and clustering.
    model_path : str or Path, optional
        Local model directory or combined .pt checkpoint; see checkpoints.py.
    components : int, default 22
        GMM K per neighborhood/segment. Nonempty patches with fewer events than
        K raise an error; choose K for your data. Empty patches return no units.
    profile : {"d1", "canonical"}, default "d1"
        Published preprocessing and threshold profile.
    enable_gate : bool, default True
        Enable event gating after the detection score threshold. False disables
        the SNR gate; thresholding, peak/NMS selection and ownership still run.
    threshold : float in [0, 1], optional
        Minimum detector sigmoid score, inclusive. None uses the model profile
        (published d1: 0.039; canonical: 0.038). This is not an amplitude threshold.
    gate_fn : callable, optional
        gate_fn(context) replaces the built-in SNR gate and requires enable_gate=True.
        context is vanillasort.gating.GateContext with read-only candidate arrays
        and segment/channel metadata. Return a boolean mask of shape (N,).
        Called once per neighborhood/segment, after thresholding and before
        ownership; skipped for empty input. Callback errors propagate unchanged.
    direct_keep : float in [0, 1], optional
        Built-in gate only: scores >= this value bypass SNR checks. None uses
        the model profile (d1: 0.045; canonical: 0.041). Independent of threshold;
        threshold >= direct_keep makes every eligible event bypass the SNR gate.
    primary_snr, secondary_snr : nonnegative float, optional
        Built-in gate only: strongest-channel and additional-channel SNR minima.
        None uses model config (published defaults: 3 and 2).
    min_secondary_channels : int in [1, 3], optional
        Built-in gate only: number of additional channels meeting secondary_snr.
        None uses model config (published default: 1).
    detector_batch, embedding_batch : int
        Inference batch sizes. CUDA OOM retries halve the affected batch.
    neighborhood_mode : {"dense", "cover"}, default "dense"
        "dense" preserves the original nearest-four neighborhoods. "cover"
        covers every contact with a deterministic subset, reducing detector
        passes but changing contexts and potentially accuracy. Both respect
        groups/shanks; use "cover" explicitly when this tradeoff is acceptable.
    target_units : int, optional
        Merge local clusters to this many nonempty units per segment. Independent
        of the per-patch GMM components. Disabled when None. Segments retain
        distinct unit IDs, so this is not a cross-segment identity constraint.
    postprocessing : dict, optional
        Fixed-K utility options. Defaults to reliable anchors and raises if fewer
        than target_units are available. on_shortfall="return_available" permits
        fewer units; anchor_policy="supplement" explicitly permits weak clusters.

    Notes
    -----
    Detection is chunked and traces are read in bounded blocks, but exact
    filtering/normalization still requires a complete four-channel segment in
    RAM. A memory check runs before reading. Event storage grows with duration.
    Segments are sorted independently with distinct unit IDs; cross-segment
    unit matching and drift tracking across neighborhoods are not implemented.
    """
    import importlib.metadata
    import json
    import time

    import numpy as np
    import psutil
    import torch
    from spikeinterface.core import BaseRecording, NumpySorting, append_sortings

    from . import __version__
    from .checkpoints import resolve_model
    from .gating import resolve_gate
    from .geometry import deduplicate_events, local_neighborhoods
    from .pipeline import run_pipeline
    from .postprocessing import InsufficientAnchorsError, _config, extract_postprocessing_waveforms, postprocess_fixed_k

    if not isinstance(recording, BaseRecording):
        raise TypeError("recording must be a SpikeInterface BaseRecording")
    fs = recording.get_sampling_frequency()
    if not np.isclose(fs, 30000.0, rtol=0, atol=1.0):
        raise ValueError("The published VanillaSort checkpoints require approximately 30 kHz; resample explicitly")
    for name, value, lower in (
        ("components", components, 2),
        ("detector_batch", detector_batch, 1),
        ("embedding_batch", embedding_batch, 1),
        ("seed", seed, 0),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < lower:
            raise ValueError(f"{name} must be an integer >= {lower}")
    if seed > 2**32 - 1:
        raise ValueError("seed must be <= 2**32 - 1")
    post_config = None
    if target_units is not None:
        if postprocessing is not None and not isinstance(postprocessing, dict):
            raise ValueError("postprocessing must be a dict of fixed-K options")
        if postprocessing is not None and "target_units" in postprocessing:
            raise ValueError("Set target_units directly, not inside postprocessing")
        post_config = _config({**(postprocessing or {}), "target_units": target_units})
    elif postprocessing is not None:
        raise ValueError("postprocessing options require target_units")
    locations, patches = local_neighborhoods(recording, mode=neighborhood_mode)
    if not patches or recording.get_num_segments() < 1:
        raise ValueError("recording must have channels and at least one segment")
    folder = Path(output_folder).expanduser().resolve() if output_folder is not None else None
    if folder is not None and folder.exists() and (not folder.is_dir() or any(folder.iterdir())):
        raise ValueError("output_folder must be new or empty; existing results are never overwritten")
    resolved_device = torch.device(("cuda:0" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
    if resolved_device.type not in ("cpu", "cuda"):
        raise ValueError("Supported devices are cpu and cuda")
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use device='cpu'")
    if verbose:
        print(
            f"VanillaSort: device={resolved_device}, {recording.get_num_channels()} channels, "
            f"{recording.get_num_segments()} segments, {len(patches)} neighborhoods per segment",
            flush=True,
        )
        print("Resolving model checkpoints (the first run may download weights)", flush=True)
    config, model_info = resolve_model(model, model_path)
    gate_options = dict(
        enable_gate=enable_gate,
        threshold=threshold,
        gate_fn=gate_fn,
        direct_keep=direct_keep,
        primary_snr=primary_snr,
        secondary_snr=secondary_snr,
        min_secondary_channels=min_secondary_channels,
    )
    gate_settings = resolve_gate(config, profile, **gate_options)
    if folder is not None:
        folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    channel_ids = np.asarray(recording.get_channel_ids())
    times_list, labels_list, main_channels, audit_runs = [], [], {}, []
    postprocess_runs = []
    next_unit = 0
    next_output_unit = 0
    old_threads = torch.get_num_threads()
    inference_threads = min(8, old_threads)
    cuda_devices = []
    if resolved_device.type == "cuda":
        cuda_devices = [resolved_device.index if resolved_device.index is not None else torch.cuda.current_device()]
    try:
        # Restore caller RNG and threading settings, even when inference fails.
        # Bound oversubscription without forcing larger CPU hosts to two threads.
        # Respect a smaller caller limit, and restore it after this sort.
        torch.set_num_threads(inference_threads)
        with torch.random.fork_rng(devices=cuda_devices):
            torch.random.default_generator.manual_seed(int(seed))
            if cuda_devices:
                with torch.cuda.device(cuda_devices[0]):
                    torch.cuda.manual_seed(int(seed))
            for segment in range(recording.get_num_segments()):
                frames = recording.get_num_samples(segment_index=segment)
                if frames < 100:
                    raise ValueError("Each segment must have at least 100 samples")
                samples_parts, labels_parts, scores_parts, patch_parts = [], [], [], []
                local_main_channels = {}
                for patch_index, (channels, owners) in enumerate(patches):
                    if verbose:
                        print(
                            f"Segment {segment + 1}/{recording.get_num_segments()}, "
                            f"neighborhood {patch_index + 1}/{len(patches)}, "
                            f"channels={channel_ids[channels].tolist()}, {time.perf_counter() - started:.1f}s elapsed",
                            flush=True,
                        )
                    required = 12 * frames * 4 * np.dtype("float32").itemsize + 512 * 2**20
                    if psutil.virtual_memory().available < required:
                        raise MemoryError(
                            f"Exact four-channel filtering needs approximately {required / 2**30:.2f} GiB "
                            "available RAM, plus event storage; normalization is not fitted per chunk"
                        )
                    raw = np.empty((frames, 4), dtype=np.float32)
                    for start in range(0, frames, 300_000):
                        stop = min(frames, start + 300_000)
                        raw[start:stop] = recording.get_traces(
                            segment_index=segment,
                            start_frame=start,
                            end_frame=stop,
                            channel_ids=channel_ids[channels],
                        )
                    if recording.has_scaleable_traces():
                        raw *= recording.get_channel_gains()[channels]
                        raw += recording.get_channel_offsets()[channels]
                    owner_local = None if len(owners) == 4 else np.flatnonzero(np.isin(channels, owners))
                    arrays, audit = run_pipeline(
                        raw,
                        locations[channels],
                        fs,
                        config,
                        resolved_device,
                        int(components),
                        int(seed),
                        profile,
                        int(detector_batch),
                        int(embedding_batch),
                        verbose=verbose,
                        owner_channels=owner_local,
                        channel_ids=channel_ids[channels],
                        segment_index=segment,
                        neighborhood_index=patch_index,
                        **gate_options,
                    )
                    del raw
                    # Keep successful CUDA batch reductions for later neighborhoods
                    # instead of repeatedly provoking the same out-of-memory error.
                    detector_batch = audit["detector"]["final_batch_size"]
                    if "embedding" in audit:
                        embedding_batch = audit["embedding"]["final_batch_size"]
                    local_units = np.unique(arrays["labels"])
                    for unit in local_units:
                        local_main_channels[next_unit + int(unit)] = channel_ids[
                            channels[arrays["main_channel_indices"][unit]]
                        ]
                    samples_parts.append(arrays["samples"])
                    labels_parts.append(arrays["labels"].astype(np.int64) + next_unit)
                    scores_parts.append(arrays["scores"])
                    patch_parts.append(np.full(len(arrays["samples"]), patch_index, dtype=np.int64))
                    next_unit += int(components)
                    audit.update(
                        segment_index=segment,
                        neighborhood_index=patch_index,
                        channel_ids=channel_ids[channels].tolist(),
                        owners=channel_ids[owners].tolist(),
                    )
                    audit_runs.append(audit)
                    if folder is not None:
                        np.savez_compressed(folder / f"segment{segment}_patch{patch_index}.npz", **arrays)
                    del arrays
                samples = np.concatenate(samples_parts)
                labels = np.concatenate(labels_parts)
                scores = np.concatenate(scores_parts)
                patch_ids = np.concatenate(patch_parts)
                keep = deduplicate_events(samples, scores, patch_ids, patches, config["refractory_samples"])
                order = np.flatnonzero(keep)[np.argsort(samples[keep], kind="stable")]
                samples, labels, scores = samples[order], labels[order], scores[order]
                if post_config is not None:
                    post_started = time.perf_counter()
                    if verbose:
                        print(f"Fixed-K postprocessing: segment {segment + 1}, target={target_units}", flush=True)
                    waves = extract_postprocessing_waveforms(
                        recording, segment, samples, config["profiles"][profile]["filter"]
                    )
                    waveform_seconds = time.perf_counter() - post_started
                    try:
                        output, post_audit = postprocess_fixed_k(
                            dict(samples=samples, labels=labels, scores=scores, waves=waves), post_config
                        )
                    except InsufficientAnchorsError as exc:
                        if folder is not None:
                            failure = {**exc.audit, "segment_index": segment}
                            (folder / f"postprocessing_segment{segment}.json").write_text(
                                json.dumps(failure, indent=2, allow_nan=False), encoding="utf-8"
                            )
                        raise
                    finally:
                        del waves
                    post_audit.update(segment_index=segment, waveform_extraction_seconds=waveform_seconds)
                    postprocess_runs.append(post_audit)
                    for unit, anchor in enumerate(post_audit["anchor_local_units"]):
                        main_channels[next_output_unit + unit] = local_main_channels[anchor]
                    samples = output["samples"][output["keep"]]
                    labels = output["labels"][output["keep"]].astype(np.int64) + next_output_unit
                    next_output_unit += post_audit["output_units"]
                    if folder is not None:
                        np.savez_compressed(folder / f"postprocessing_segment{segment}.npz", **output)
                        (folder / f"postprocessing_segment{segment}.json").write_text(
                            json.dumps(post_audit, indent=2, allow_nan=False), encoding="utf-8"
                        )
                else:
                    main_channels.update(local_main_channels)
                times_list.append(samples)
                labels_list.append(labels)
    finally:
        torch.set_num_threads(old_threads)
    # Build single segments first: NumpySorting infers segment count from the
    # last spike and would otherwise lose empty trailing segments.
    unit_ids = np.unique(np.concatenate(labels_list))
    segments = [
        NumpySorting.from_unit_dict({unit: times[labels == unit] for unit in unit_ids}, sampling_frequency=fs)
        for times, labels in zip(times_list, labels_list)
    ]
    sorting = segments[0] if len(segments) == 1 else append_sortings(segments)
    sorting.set_property(
        "main_channel_id", np.asarray([main_channels[int(u)] for u in sorting.unit_ids], dtype=channel_ids.dtype)
    )
    sorting.annotate(vanillasort_version=__version__, model_version=model_info["version"])
    sorting.annotate(detection_gate=gate_settings)
    if post_config is not None:
        sorting.annotate(fixed_k_postprocessing=postprocess_runs)
    if folder is not None:
        sorting.save(folder=folder / "sorting")
        np.savez_compressed(
            folder / "events.npz",
            sample_index=np.concatenate(times_list),
            unit_id=np.concatenate(labels_list),
            segment_index=np.concatenate(
                [np.full(len(times), segment, dtype=np.int64) for segment, times in enumerate(times_list)]
            ),
        )
        metadata = dict(
            version=__version__,
            model=model_info,
            sampling_frequency=fs,
            seed=int(seed),
            components=int(components),
            neighborhood_mode=neighborhood_mode,
            target_units=post_config["target_units"] if post_config is not None else None,
            postprocessing_runs=postprocess_runs,
            profile=profile,
            gating=gate_settings,
            device=str(resolved_device),
            torch_num_threads=inference_threads,
            channel_ids=channel_ids.tolist(),
            channel_locations=locations.tolist(),
            segments=recording.get_num_segments(),
            runs=audit_runs,
            elapsed_seconds=time.perf_counter() - started,
            config=config,
            versions={
                name: importlib.metadata.version(name)
                for name in ("torch", "numpy", "scipy", "scikit-learn", "spikeinterface")
            },
        )
        (folder / "run.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    sorting.register_recording(recording)
    return sorting
