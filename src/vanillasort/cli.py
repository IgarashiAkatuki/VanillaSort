"""Command line interface for the published four-channel workflow."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import psutil
import torch

from .checkpoints import resolve_model, file_hash, verify_bundle
from .pipeline import run_pipeline


def read_input(path, coords_path=None, fs_hz=30000.0):
    """NPZ: traces[T,4], coords[4,2], optional fs_hz. NPY plus --coords also works."""
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as z:
            raw = z["traces"].copy()
            coords = z["coords"].copy() if "coords" in z else None
            if "fs_hz" in z:
                fs_hz = float(np.asarray(z["fs_hz"]).item())
    elif path.suffix.lower() == ".npy":
        raw = np.load(path, mmap_mode="r", allow_pickle=False)
        coords = None
    else:
        raise ValueError("Use .npz or .npy input; export raw [time,4] traces first")
    if coords_path is not None:
        coords = np.load(coords_path, allow_pickle=False)
    if coords is None:
        raise ValueError("Supply coords[4,2] in NPZ or --coords coords.npy; channel order must match traces")
    return raw, coords, fs_hz


def synthetic_example():
    """Reproducible connectivity test, not a performance benchmark or real data."""
    rng = np.random.default_rng(42)
    fs = 30000.0
    raw = rng.normal(0, 1, (60000, 4)).astype(np.float32)
    t = np.arange(-30, 30)
    spike = -np.exp(-((t / 3.0) ** 2)) + 0.35 * np.exp(-(((t - 9) / 6.0) ** 2))
    for j, center in enumerate(range(100, len(raw) - 100, 150)):
        amp = np.roll(np.array([8.0, 5.0, 2.0, 1.0], np.float32), j % 4)
        raw[center - 30 : center + 30] += (spike[:, None] * amp).astype(np.float32)
    coords = np.array([[0, 0], [20, 0], [0, 20], [20, 20]], dtype=np.float32)
    return raw, coords, fs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="NPZ with traces, coords, fs_hz; or NPY with --coords")
    parser.add_argument("--coords", type=Path, help="Physical channel coordinates NPY [4,2], micrometres")
    parser.add_argument("--fs", type=float, default=30000.0)
    parser.add_argument(
        "--seconds", type=float, help="Explicit prefix length; do not use for full-record metric reproduction"
    )
    parser.add_argument("--components", type=int, help="Explicit GMM K; Hybrid rec11/12:22, others:24")
    parser.add_argument("--seed", type=int, default=30, help="One fixed historical restart; never GT-selected")
    parser.add_argument("--profile", choices=["d1", "canonical"], default="d1")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:0")
    parser.add_argument("--detector-batch", type=int, default=8)
    parser.add_argument("--embedding-batch", type=int, default=128)
    parser.add_argument(
        "--gpu-fraction", type=float, default=0.4, help="CUDA allocator upper limit; OOM halves a batch"
    )
    parser.add_argument("--detect-only", action="store_true")
    parser.add_argument("--self-test", action="store_true", help="Run synthetic 2s end-to-end, K=4")
    parser.add_argument("--model-path", type=Path, help="Local combined checkpoint or model directory (offline)")
    parser.add_argument("--verify-only", action="store_true", help="Download/cache and verify default model SHA256")
    parser.add_argument("--output", type=Path, default=Path("output"))
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify_bundle(), ensure_ascii=False, indent=2))
        return 0
    if not args.self_test and args.input is None:
        parser.error("Supply --input or --self-test")
    if not args.self_test and not args.detect_only and (args.components is None or args.components < 2):
        parser.error("Supply explicit --components >=2 (no ground-truth or automatic K selection)")
    if min(args.detector_batch, args.embedding_batch) < 1 or not 0 < args.gpu_fraction <= 1:
        parser.error("Batch sizes must be positive; --gpu-fraction must be in (0,1]")
    if args.self_test and args.input is not None:
        parser.error("--self-test and --input are mutually exclusive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output folder must be empty/new; existing results are never overwritten")
    device = torch.device(("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if device.type not in ("cpu", "cuda"):
        parser.error("Supported devices are cpu/cuda")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA is unavailable; use --device cpu")
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction, device)
        torch.cuda.reset_peak_memory_stats(device)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    # Preserve the historical FP32 execution; no new AMP/TF32 policy is introduced.
    config, model_info = resolve_model(model_path=args.model_path)
    started = time.perf_counter()
    raw, coords, fs = synthetic_example() if args.self_test else read_input(args.input, args.coords, args.fs)
    if args.seconds is not None:
        frames = int(round(args.seconds * fs))
        if frames <= 0 or frames > len(raw):
            parser.error("--seconds must be positive and no longer than the supplied recording")
        raw = raw[:frames]
    required = 12 * np.asarray(raw).nbytes + 512 * 2**20
    if psutil.virtual_memory().available < required:
        raise MemoryError(
            f"Full-record filtering requires approximately {required/2**30:.2f} GiB available RAM; no silent chunk-wise normalization"
        )
    arrays, audit = run_pipeline(
        raw,
        coords,
        fs,
        config,
        device,
        4 if args.self_test else args.components,
        args.seed,
        args.profile,
        args.detector_batch,
        args.embedding_batch,
        args.detect_only,
    )
    audit.update(
        elapsed_seconds=time.perf_counter() - started,
        device=str(device),
        synthetic_smoke_test=args.self_test,
        config=config,
        model=model_info,
        input_path=str(args.input.resolve()) if args.input else None,
        versions={
            n: importlib.metadata.version(n)
            for n in ("torch", "numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl", "psutil")
        },
        torch_cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        checkpoint_sha256={n: file_hash(config[n]) for n in ("detector_checkpoint", "huidurep_checkpoint")},
    )
    if device.type == "cuda":
        audit["peak_gpu_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
        audit["peak_gpu_reserved_gib"] = torch.cuda.max_memory_reserved(device) / 2**30
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "events.npz", **arrays)
    (args.output / "run.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f'Done: {args.output.resolve()} ({audit["elapsed_seconds"]:.1f}s)', flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
