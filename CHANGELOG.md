# Changelog

## 0.2.1 — 2026-10-19

- Expose detector selection settings in `vanillasort.sort()`: `enable_gate`,
  `threshold`, `direct_keep`, `primary_snr`, `secondary_snr`, and
  `min_secondary_channels`. Add corresponding CLI flags, including `--no-gate`.
  Omitted thresholds retain the loaded model's defaults. Disabling the SNR gate
  retains score thresholding, peak/NMS selection, and channel ownership checks.
- Add Python `gate_fn(context)` callbacks through the independent
  [gating utility](src/vanillasort/gating.py). A custom gate replaces the built-in
  SNR and direct-acceptance rule after score thresholding. `GateContext` provides
  read-only candidate arrays and channel/segment metadata; callbacks must return
  a boolean keep mask. Empty inputs skip the callback, invalid masks fail
  explicitly, and callback exceptions propagate to the caller.
- Add the standalone [fixed-K utility](src/vanillasort/postprocessing.py) and
  optional `target_units`/`postprocessing` arguments to the recording API. Merge
  local clusters after cross-neighborhood event deduplication, using common-channel
  waveforms. `components` remains the GMM size per neighborhood; `target_units`
  constrains nonempty output units per segment, with separate identities across
  segments.
- Handle missing reliable anchors, empty detections, degenerate templates,
  input noise labels, invalid inputs, and K=1 in fixed-K postprocessing.
  Insufficient reliable anchors raise `InsufficientAnchorsError` with diagnostics
  by default. `on_shortfall="return_available"` explicitly permits fewer units;
  `anchor_policy="supplement"` permits observed weak/duplicate clusters and marks
  degraded results. Missing units are never fabricated, and events are
  deduplicated within each output unit.
- Add opt-in `neighborhood_mode="cover"` to reduce repeated full-record detector
  passes using a covering subset of nearest-four neighborhoods. Every channel
  remains an input and owns exactly one patch, respecting group/shank boundaries.
  Keep `"dense"` as the default: sparser contexts reduced sorting accuracy in
  validation recordings despite reducing detector runtime.
- Save resolved gate settings, selection counts, gating time, and fixed-K
  diagnostics in run outputs. Expose gate settings and postprocessing diagnostics
  as sorting annotations, and save per-segment postprocessing arrays/JSON when an
  output folder is supplied. Preserve the model configuration; custom callback
  logs contain the callable name, so reproduction also requires its code and
  captured parameters. Document the new options and examples in both READMEs.

Existing calls retain their default gate, thresholds, dense neighborhoods, and
disabled fixed-K postprocessing. `threshold` and `direct_keep` are independent:
when `threshold >= direct_keep`, every eligible event bypasses the built-in SNR
check. Custom functions require `enable_gate=True` and are available through the
Python API only. SpikeInterface `run_sorter()` wrappers must expose the new API
options separately before forwarding them.

Validation: all 16 tests pass. With real checkpoints, default inference and an
equivalent custom gate match the frozen 0.2.0 arrays exactly. An eight-channel,
60-second run with `threshold=0.139`, gating disabled, and `target_units=10`
matches the previous output exactly: 10 units and 6,526 spikes. These are
regression checks, not a general accuracy or runtime guarantee.

## 0.2.0 — 2026-10-18

- Unify research development and the Python package on `main`, retaining the
  `src/vanillasort/` layout, public `sort()` API, CLI, and pinned model downloads.
- Evaluate long rotary-detector inference with local attention tiles. The
  receptive field, checkpoint format, and training path are preserved.
- Report detection and embedding progress, neighborhood counters, and per-stage
  timings in `run.json`.
- Use up to eight PyTorch CPU threads for API sorting, respecting and restoring
  the caller's limit. Reuse reduced CUDA batch sizes across neighborhoods.
- Update bilingual documentation and add a reusable SpikeInterface reproduction
  script and [profiling report](docs/performance-pr4832.md).

On one complete 60-second, four-channel neighborhood, measured CPU runtime fell
from 96.31 s to 54.12 s with two threads, and to 33.27 s with eight threads.
Spike times, labels, and features matched exactly; detector scores differed by
at most 1.20e-7. The report records the hardware, conditions, and partial
64-channel verification. CUDA performance was not measured.

## 0.1.0 — 2026-10-06

Initial PyPI release with the SpikeInterface recording API, command-line
inference, and SHA-256-verified pretrained checkpoints cached from Hugging Face.
