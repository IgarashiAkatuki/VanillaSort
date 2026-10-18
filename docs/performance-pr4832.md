# SpikeInterface PR #4832: progress and runtime investigation

Measured on 2026-10-18 for the [maintainer's example](https://github.com/SpikeInterface/spikeinterface/pull/4832#issuecomment-6219508279).

## Reproduction

The input is `generate_ground_truth_recording(num_channels=64, durations=[60],
sampling_frequency=30_000, seed=0)`, saved once with `n_jobs=-1`. The fixed
generation seed allows both implementations to use identical traces. All
sorter parameters retain their defaults. The published checkpoints were
loaded from a local copy to exclude download time; their SHA-256 values match
the pinned Hugging Face release.

Environment: macOS arm64, 24 GiB RAM, Python 3.12.13, PyTorch 2.14.1,
NumPy 2.5.4, SciPy 1.18.1, scikit-learn 1.9.1, and the local SpikeInterface
0.105.1 checkout. This machine has no CUDA device. The archived VanillaSort
source at `0c3d28c` was compared with installed PyPI 0.1.0: every Python and
configuration file matched.

The original wrapper reproduces the missing-progress symptom: its terminal
output contains the already-filtered warning while inference continues in the
background. Package progress is present in `sorter_output/vanillasort.log`.

## Measured bottleneck

The default two-column probe generates **62 distinct overlapping four-channel
neighborhoods**. Each neighborhood runs detection over all 1,800,000 samples:
720 detector chunks, grouped into 90 batches at the default batch size of 8.
The complete adapter therefore evaluates 44,640 chunks across 62 passes.

The first baseline neighborhood took 100.12 seconds in the instrumented run:
96.47 seconds in detection, 2.92 seconds in embedding, and 0.13 seconds in
clustering. Detector and embedding checkpoint loading together took about
0.12 seconds. A CPU operator profile of one detector batch attributed 57.5%
of self CPU time to attention and 28.0% to dense linear multiplication.

The detector's mask permits only a local +/-256-sample window, but its
original attention call spans a full 2500-by-2500 matrix. The repeated
neighborhood passes and this dense attention work explain the observed CPU
runtime. Model downloads, filtering, and clustering were not the bottleneck
in this reproduction. A small GPU may select different kernels; its exact
runtime has not been measured here.

## Changes

- The SpikeInterface wrapper tees stdout to the console when `verbose=True`
  and always retains its log. Quiet execution and stdout restoration after
  exceptions are covered by the existing integration test.
- Detector and embedding batches report flushed progress about every five
  seconds. Output includes the device, segment and neighborhood counters,
  elapsed time, and stage estimates. `runs[].timings_seconds` in `run.json`
  records preprocessing, detection, embedding, clustering, and total time.
- Long rotary inference uses 256-query tiles, retaining all permitted keys
  in each tile's halo. RoPE positions, receptive fields, masks, checkpoints,
  and FP32 policy are unchanged. Training and gradient-enabled evaluation
  retain the original dense attention path.
- The recording API uses up to eight PyTorch threads, respecting a smaller
  caller limit and restoring it afterward. Successful CUDA batch reductions
  persist across subsequent neighborhoods and segments in the same sort.

## Isolated timing and numerical checks

Both long runs were suspended during the following sequential measurements
of the same complete 60-second first neighborhood. Input, weights, ownership
rules, K=22, seed=0, and batch sizes were identical.

| Implementation | PyTorch threads | Time | Speed relative to original |
| --- | ---: | ---: | ---: |
| Original dense attention | 2 | 96.31 s | 1.00x |
| Local attention tiles | 2 | 54.12 s | 1.78x |
| Local attention tiles | 8 | 33.27 s | 2.90x |

All three produced the same 1,760 sortable events. Detection times, sorting
times, SNRs, embeddings, amplitude features, initial/final labels, templates,
GMM parameters, and main channels were exactly equal. Only detector scores
differed, by at most 1.20e-7. The two-second D1 and canonical fixtures also
retained their 446 and 415 sorted events and identical core arrays.

A separate real-checkpoint test forcing the CPU math attention backend
produced exactly equal logits for one 2500-sample chunk. This is a numerical
check of that backend, not a CUDA performance measurement.

Eight package tests and ten SpikeInterface integration, discovery, and
dependency tests passed; one dependency test was skipped. Tests include
causal/padded tile boundaries, gradient compatibility, visible and quiet
stdout behavior, and simulated CUDA out-of-memory retries.

## Partial full-recording verification

Both full runs were stopped at the user's request. The original completed
12/62 neighborhoods and the optimized implementation completed 23/62.
The 12 completed pairs contain 7,375 sortable events before cross-neighborhood
deduplication. Every core array listed above matches exactly in those pairs;
the maximum detector-score difference is 4.77e-7. Completed per-neighborhood
artifacts and logs were retained.

Neither run reached final cross-neighborhood deduplication and sorting export.
The remaining neighborhoods and complete 64-channel sorting output therefore
remain unverified. These two runs overlapped and were paused for the isolated
measurements, so their wall times are not a controlled speed comparison.

Local implementation commits are VanillaSort `db4b4b2` and SpikeInterface
`6b50a3d1e`. The VanillaSort changes are included in version 0.2.0;
the wrapper changes are maintained in the SpikeInterface contribution.

## Re-run

Install both development checkouts, then run:

```bash
python examples/reproduce_si_4832.py output/si4832-run --device cpu
```

Use `--device cuda` on an NVIDIA machine. `--model-path` can select a local
copy of the published weights. The script retains the cached input and
requires a new result folder for each run. The larger-probe adapter still
runs each neighborhood independently; this change preserves that algorithm
and does not establish sorting accuracy on arbitrary larger probes.
