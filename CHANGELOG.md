# Changelog

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
