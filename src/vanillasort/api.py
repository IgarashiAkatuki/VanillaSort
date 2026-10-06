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
    detector_batch=8,
    embedding_batch=128,
) -> BaseSorting:
    """Sort a raw 30 kHz recording and return a SpikeInterface BaseSorting.

    Parameters
    ----------
    recording : spikeinterface.core.BaseRecording
        Raw traces on a common voltage scale, with finite 2D channel locations
        in micrometres. Each group/shank must contain at least four channels.
        Larger groups use experimental overlapping nearest-four neighborhoods.
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
    detector_batch, embedding_batch : int
        Inference batch sizes. CUDA OOM retries halve the affected batch.

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
    from .geometry import deduplicate_events, local_neighborhoods
    from .pipeline import run_pipeline

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
    locations, patches = local_neighborhoods(recording)
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
    config, model_info = resolve_model(model, model_path)
    if profile not in config["profiles"]:
        raise ValueError(f"Unknown profile {profile!r}")
    if folder is not None:
        folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    channel_ids = np.asarray(recording.get_channel_ids())
    times_list, labels_list, main_channels, audit_runs = [], [], {}, []
    next_unit = 0
    old_threads = torch.get_num_threads()
    cuda_devices = []
    if resolved_device.type == "cuda":
        cuda_devices = [resolved_device.index if resolved_device.index is not None else torch.cuda.current_device()]
    try:
        # Restore caller RNG and threading settings, even when inference fails.
        torch.set_num_threads(2)
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
                for patch_index, (channels, owners) in enumerate(patches):
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
                    )
                    del raw
                    local_units = np.unique(arrays["labels"])
                    for unit in local_units:
                        main_channels[next_unit + int(unit)] = channel_ids[
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
                chronological = np.argsort(samples[keep], kind="stable")
                times_list.append(samples[keep][chronological])
                labels_list.append(labels[keep][chronological])
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
            profile=profile,
            device=str(resolved_device),
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
