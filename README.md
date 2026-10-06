# VanillaSort

**English** | [简体中文](README.zh-CN.md)

VanillaSort is a spike-sorting pipeline for extracellular recordings. It combines **VanillaDet** spike detection, **HuiduRep** waveform representations, and **VanillaCluster** clustering to produce spike times and putative neuronal-unit assignments.

This installable Python package provides a SpikeInterface API, a command-line runner, and pretrained checkpoints for **four-channel recordings sampled at 30 kHz**. Larger 2D probes use an experimental nearest-four-neighbor adapter.

**Paper:** Zishuo Feng and Feng Cao, [*Spike Sorting with VanillaSort*](https://www.biorxiv.org/content/10.64898/2026.09.18.752552), bioRxiv, 2026. **You must cite this paper when using VanillaSort.** [BibTeX](#citation)

[Results](#results) · [Installation](#installation) · [Quick start](#quick-start) · [Your data](#sort-your-data) · [Outputs](#outputs) · [Configuration](#configuration) · [Method](#method)

## Results

The [paper's Tables 1–2](https://www.biorxiv.org/content/10.64898/2026.09.18.752552) report results on **Hybrid Janelia Static and Drift**, with nine recordings per subset and ground-truth units with SNR ≥ 3. Sorting accuracy is `TP / (TP + FP + FN)`, with event matching within ±6 samples. Values below are **mean ± SEM**; higher is better.

| Spike sorter | Static accuracy | Drift accuracy |
| --- | ---: | ---: |
| HerdingSpikes2 | 0.35 ± 0.01 | 0.29 ± 0.01 |
| IronClust | 0.57 ± 0.04 | 0.54 ± 0.03 |
| JRClust | 0.47 ± 0.04 | 0.35 ± 0.03 |
| KiloSort | 0.60 ± 0.02 | 0.51 ± 0.02 |
| KiloSort2 | 0.39 ± 0.03 | 0.30 ± 0.02 |
| KiloSort4 | 0.40 ± 0.03 | 0.34 ± 0.02 |
| MountainSort4 | 0.59 ± 0.02 | 0.36 ± 0.02 |
| MountainSort5 | 0.40 ± 0.06 | 0.33 ± 0.04 |
| SpykingCircus | 0.57 ± 0.01 | 0.48 ± 0.02 |
| Tridesclous | 0.54 ± 0.03 | 0.37 ± 0.02 |
| SimSort | 0.62 ± 0.04 | 0.56 ± 0.03 |
| HuiduRep (without DAE) | 0.69 ± 0.02 | 0.56 ± 0.02 |
| HuiduRep (with DAE) | 0.70 ± 0.02 | 0.60 ± 0.02 |
| **VanillaSort (without DAE)** | **0.73 ± 0.02** | 0.61 ± 0.02 |
| **VanillaSort (with DAE)** | **0.73 ± 0.01** | **0.64 ± 0.02** |

DAE denotes a denoising autoencoder. Other-tool scores are the SpikeForest or original-publication results used in the paper.

- **Detection:** VanillaDet achieves 0.74/0.71 accuracy on Static/Drift, compared with SimSort's 0.72/0.68 and amplitude thresholding's 0.61/0.60.
- **Sorting:** VanillaSort improves accuracy over the corresponding HuiduRep baselines by **3–4 percentage points on Static** and **4–5 points on Drift** (paired Wilcoxon test, *p* < 0.05).

## Installation

Requirements: **Python 3.10+**, **PyTorch 2.7.0+**, and **CUDA 12.8+** with a compatible NVIDIA driver for GPU execution. **Use the latest compatible stable releases whenever possible.**

Create a virtual environment:

```bash
git clone https://github.com/IgarashiAkatuki/VanillaSort.git
cd VanillaSort
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`.

For GPU execution, select **Stable** and the newest compatible CUDA version **12.8 or later** in the [PyTorch installation selector](https://pytorch.org/get-started/locally/). Use its wheel index in the installation command. The table includes a CUDA 12.8 example:

| Platform | Command |
| --- | --- |
| Linux / Windows, CPU | `python -m pip install --upgrade "torch>=2.7.0" --index-url https://download.pytorch.org/whl/cpu` |
| Linux / Windows, NVIDIA GPU, CUDA 12.8 example | `python -m pip install --upgrade "torch>=2.7.0" --index-url https://download.pytorch.org/whl/cu128` |
| macOS, Apple silicon, CPU | `python -m pip install --upgrade "torch>=2.7.0"` |

Install the package (runtime dependencies are declared in `pyproject.toml`):

```bash
python -m pip install -e .
```

[`requirements.txt`](requirements.txt) records historical numerical versions for comparison; it is not the package installation command. Each run records its installed versions in `run.json`. For development, use `pip install -e ".[dev]"` and `pytest`. The model classes already include their training operations, so no separate training dependencies are required.

The model checkpoints (about 36 MiB total) are hosted at [Kohaku2580/VanillaSort on Hugging Face](https://huggingface.co/Kohaku2580/VanillaSort). They are downloaded on the first model load and cached locally. The wheel and source distribution contain no `.pt` files; ordinary import and the automated tests do not download weights. The `vanillasort` command and `python -m vanillasort` work outside the repository; `python run.py` remains a compatibility entry point.

## Python / SpikeInterface API

```python
import vanillasort
from spikeinterface.core import BaseSorting, load

# recording is an existing BaseRecording with probe/channel locations.
sorting = vanillasort.sort(recording)
assert isinstance(sorting, BaseSorting)
print(vanillasort.__version__)

sorting = vanillasort.sort(
    recording,
    output_folder="output/recording",
    model="default",
    device="auto",  # CUDA when available, otherwise CPU
    seed=0,
    components=22,  # choose K for your data; there is no automatic K estimation
    verbose=True,
)
restored = load("output/recording/sorting")
```

The API accepts raw recordings at 30,000 Hz (tolerance 1 Hz), at least 100 samples per segment, and finite 2D locations in micrometres. Stored channel gains/offsets are applied when available; otherwise traces must already share a voltage scale. Filtering is internal. Unit spike trains contain zero-based sample indices, with the original segment index. `main_channel_id` uses the original recording channel IDs and the largest median peak-to-peak amplitude across assigned events.

The remaining options are `model_path=None`, `profile="d1"` (or `"canonical"`), `detector_batch=8` and `embedding_batch=128`. `components=22` is a historical preset, **not an estimate of the number of neurons**. A nonempty neighborhood with fewer than K events raises an error; empty detections produce an empty sorting. The API seed defaults to 0; the historical CLI retains seed 30.

With `output_folder=None`, no files are written. Otherwise use a new/empty folder: `sorting/` is reloadable with SpikeInterface, `events.npz` contains aligned `sample_index`, `unit_id`, and `segment_index`, `segmentN_patchM.npz` contains detailed local pipeline arrays, and `run.json` records parameters, geometry, model hashes and diagnostics. The registered recording stays in memory and is not copied into `sorting/`; call `restored.register_recording(recording)` if needed. The CLI output format described below is preserved.

### Geometry and recording length

Exactly four channels in a group retain their original order and published inference behavior. Larger groups use the four nearest real contacts around each channel, preserve recording order within each unique neighborhood, and assign detections to the neighborhood owning their strongest SNR channel. Score-priority suppression within 12 samples removes duplicates across intersecting neighborhoods. Channel groups and probe/shank boundaries are respected. The existing waveform repetition/cropping to 11 HuiduRep channels is unchanged.

VanillaDet has **no missing-channel mask**. Groups/shanks with fewer than four contacts and 3D probes are rejected. The larger-probe adapter is experimental: units can split when their strongest channel moves between neighborhoods; overlapping neighborhoods can suppress nearby simultaneous events. It does not implement drift correction or cross-neighborhood unit merging. Segments are clustered independently with distinct unit IDs; no cross-segment unit matching is assumed.

Traces are read in blocks of at most 300,000 samples for one four-channel neighborhood at a time. Detection retains the original 2,500-sample chunks and global refractory suppression; embeddings are batched. Exact whole-segment filtering and median/MAD normalization still require RAM proportional to the segment duration, with a preflight memory check. Waveforms/features also grow with event count. This is not an out-of-core sorter. CUDA OOM handling halves inference batches, as in the original runner.

### Model checkpoints

`model="default"` resolves `hybrid-janelia-2026.09` from [Hugging Face](https://huggingface.co/Kohaku2580/VanillaSort/tree/dbaa0cf5d14a737d494af0fafff402f62453c6ee), pinned to commit `dbaa0cf5d14a737d494af0fafff402f62453c6ee`. Both weights are byte-identical to the [original repository checkpoints](https://github.com/IgarashiAkatuki/VanillaSort/tree/00a9ef06e88a955406ad2902d7bc56617f0afd2e). The repository ID, revision, filenames and SHA-256 hashes are centralized in [`configs/default_model.json`](src/vanillasort/configs/default_model.json); architecture and inference settings live in [`configs/default.json`](src/vanillasort/configs/default.json).

The first `sort()` or CLI inference downloads the weights using `huggingface_hub`; every load verifies the hashes. Subsequent loads check the pinned local snapshot first and need no network request. The standard Hugging Face cache is used by default (`HF_HOME` / `HF_HUB_CACHE` are supported), or set `VANILLASORT_MODEL_CACHE` to an alternate **Hub cache root**. `HF_HUB_OFFLINE=1` supports offline use after the cache has been populated. An empty offline cache raises a clear error; supplying `model_path` is another fully offline option. Public default weights require no token. `python -m vanillasort --verify-only` populates/checks the default cache; the two-second CLI self-test also downloads weights if needed.

```python
sorting = vanillasort.sort(recording, model_path="/path/to/model.pt", components=22)
```

A combined `.pt` must contain `detector_state_dict` and `huidurep_state_dict`; optional `config` overrides the default configuration. Alternatively, `model_path` can name a directory with `config.json` and both checkpoint files. That JSON specifies `detector_checkpoint` and `huidurep_checkpoint`, relative to the directory. Individual checkpoint files accept raw state dicts, `state_dict` or `model_state_dict` containers, and `module.` prefixes. Loading uses `weights_only=True` and strict state-dict matching.

The canonical model classes are shared by training and inference:

```python
import torch
from vanillasort.models import VanillaDet, HuiduRep

# Use architectures matching your training configuration.
# Save a compatible combined checkpoint from trained model instances:
# torch.save({"detector_state_dict": detector.state_dict(),
#             "huidurep_state_dict": encoder.state_dict(),
#             "config": config}, "model.pt")
```

This checkout contains the published inference bundle and model training methods, but no training driver or training dataset. Paper profiles, K settings, seed controls and numerical reference versions remain documented below; package installation does not add a training recipe.

## Quick start

Run the complete pipeline on the built-in two-second synthetic recording with four mixture components:

```bash
python run.py --self-test --device cpu --output output/self-test
```

This checks detection, waveform encoding, clustering, and result export. A successful run prints `Done:` and saves `events.npz` and `run.json` in `output/self-test/`.

For your recordings, `--device auto` selects CUDA when available and CPU otherwise. Use a new or empty output directory for each run.

## Sort your data

### Prepare the input

Save a NumPy `.npz` archive containing:

| Key | Shape | Content |
| --- | --- | --- |
| `traces` | `(T, 4)` | Raw voltage samples, with time along the first axis. `float32` is recommended. |
| `coords` | `(4, 2)` | Physical channel coordinates in micrometres, in the same channel order as `traces`. |
| `fs_hz` | Scalar | Sampling frequency in Hz; use `30000.0`. |

Use finite values, at least 100 time samples, and a common voltage scale across channels. Filtering and normalization are applied by the pipeline. For a larger probe, export a four-channel group and its matching coordinates.

For example, package existing trace and coordinate arrays:

```python
import numpy as np

traces = np.load("traces.npy")  # (T, 4)
coords = np.load("coords.npy")  # (4, 2), micrometres

np.savez(
    "recording.npz",
    traces=traces.astype(np.float32),
    coords=coords.astype(np.float32),
    fs_hz=30000.0,
)
```

### Run sorting

```bash
python run.py --input recording.npz --components 22 --output output/recording-k22
```

Set `--components` to the number of Gaussian mixture components, **K**, chosen for your recording. K must be at least 2, and the number of sortable events must be at least K. The bundled Hybrid Janelia settings use **22** for recordings **11/12** and **24** for **21/22/31/32**.

You can also supply separate `.npy` files:

```bash
python run.py --input traces.npy --coords coords.npy --fs 30000 --components 22 --output output/recording-npy
```

An NPZ `fs_hz` value takes precedence over `--fs`; `--coords` takes precedence over coordinates stored in the NPZ.

To export spike detections alone:

```bash
python run.py --input recording.npz --detect-only --output output/detections
```

## Outputs

Each run writes two files:

| File | Content |
| --- | --- |
| `events.npz` | Event times, detector scores, and, for sorting runs, labels and model outputs. |
| `run.json` | Configuration, sampling rate, event counts, normalization statistics, runtime, package versions, checkpoint SHA-256 hashes, and processing diagnostics. |

The following arrays are saved for a completed clustering run. Here, **D** is the number of accepted detections, **N** is the number of sortable events, and **K** is the requested component count.

| Array | Shape | Meaning |
| --- | --- | --- |
| `detection_samples` | `(D,)` | Accepted detection times as zero-based sample indices. |
| `detection_scores` | `(D,)` | Detector sigmoid scores. |
| `detection_snrs` | `(D, 4)` | Per-event, per-channel SNR estimates. |
| `samples` | `(N,)` | Sample indices of events retained for sorting. |
| `scores` | `(N,)` | Detector scores aligned with `samples`. |
| `labels`, `initial_labels` | `(N,)` each | Final assignments and initial GMM assignments, indexed from `0` to `K - 1`. |
| `features` | `(N, 32)` | HuiduRep embeddings standardized across events in the recording. |
| `auxiliary` | `(N, 3)` | Standardized relative-amplitude contrasts. |
| `templates` | `(2, K, 60, 4)` | Waveform templates indexed by target fold, component, sample, and channel. |
| `core_counts` | `(2, K)` | Core-event counts used to determine template availability. |
| `means` | `(K, 35)` | GMM component means in the combined feature space. |
| `covariances`, `weights` | `(K, 35, 35)`, `(K,)` | GMM full covariance matrices and mixture weights. |

Sorting retains detections satisfying `30 <= sample < T - 30` and extracts waveform offsets `-30` through `+29`. This accounts for the difference between D and N. Detection-only runs save the three `detection_*` arrays.

Read spike times in seconds and select a unit:

```python
import json
from pathlib import Path
import numpy as np

output = Path("output/recording-k22")
metadata = json.loads((output / "run.json").read_text(encoding="utf-8"))
with np.load(output / "events.npz") as events:
    spike_times_s = events["samples"] / metadata["fs_hz"]
    unit_ids = events["labels"]

unit_0_times_s = spike_times_s[unit_ids == 0]
units, spike_counts = np.unique(unit_ids, return_counts=True)
print(dict(zip(units.tolist(), spike_counts.tolist())))
```

Times are relative to the start of the supplied recording. Review putative units using their waveforms, firing rates, and inter-spike intervals. The `clustering` section of `run.json` records GMM convergence, warnings, template construction, and changes made during assignment refinement.

## Configuration

Common command-line options:

| Option | Default | Purpose |
| --- | --- | --- |
| `--components K` | Required for sorting your data | Number of GMM components. |
| `--model-path PATH` | Hugging Face default model | Use a local checkpoint bundle or directory. |
| `--profile d1` | `d1` | Select `d1` or `canonical` preprocessing. |
| `--seed 30` | `30` | Random seed for the run and its single GMM initialization. |
| `--device auto` | `auto` | Select `auto`, `cpu`, `cuda`, or a device such as `cuda:0`. |
| `--seconds 10` | Entire recording | Process the first 10 seconds; the value must fit within the recording. |
| `--detector-batch 8` | `8` | Number of 2,500-sample chunks per detector batch. |
| `--embedding-batch 128` | `128` | Number of waveforms per HuiduRep batch. |
| `--gpu-fraction 0.4` | `0.4` | Limit the PyTorch CUDA allocator to this fraction of device memory. |
| `--output PATH` | `output` | New or empty result directory. |

CUDA out-of-memory handling halves the affected inference batch and retries. Filtering and normalization use the entire selected recording in host memory. Before processing, the runner checks for approximately `12 × input array bytes + 512 MiB` of available RAM; waveform and feature storage also grows with the event count.

The profiles and model settings are defined in [`configs/default.json`](src/vanillasort/configs/default.json):

| Setting | `d1` (default) | `canonical` |
| --- | --- | --- |
| Band-pass filter | 300–5,000 Hz, order 5 | 200–6,000 Hz, order 3 |
| Notch filter | 60 Hz | Off |
| Detection threshold | 0.039 | 0.038 |
| Direct-acceptance threshold | 0.045 | 0.041 |
| Waveform source | Filtered voltage | Filtered voltage with per-channel median/MAD normalization |

Both profiles use per-channel median/MAD normalization for detector input. Event-SNR gating uses the canonical 200–6,000 Hz signal. Processing a prefix with `--seconds` also fits normalization and clustering to that prefix. Keep `run.json` with your results to record these choices.

## Method

1. **VanillaDet — detect spikes.** A convolutional frontend and a six-layer local Transformer produce sample-level scores. The default model uses 256-dimensional hidden states, four attention heads, rotary positional encoding, and L2-normalized queries and keys. Peak selection applies a 12-sample exclusion radius. Events at or above the direct-acceptance threshold are retained; events between the two thresholds require a primary-channel SNR of at least 3 and an SNR of at least 2 on another channel.
2. **HuiduRep — encode waveforms.** Each event contributes a 60-sample, four-channel waveform. Preprocessing standardizes each channel over events and time, interpolates to 90 samples, and repeats/crops channels to 11 inputs. HuiduRep applies internal denoising and produces a 32-dimensional representation, which is standardized across the recording's events.
3. **VanillaCluster — assign units.** Three relative-amplitude features are formed from per-channel peak-to-peak amplitude fractions, projected onto a Helmert contrast basis, and standardized. A full-covariance GMM clusters the resulting 35-dimensional vectors. One template-residual refinement pass then updates assignments among the top three candidate components.

Template refinement splits events into alternating one-second blocks. For each target fold, templates are built from the opposite fold using events with GMM posterior probability of at least 0.9. Each core pool is capped at 512 events, then its lowest-scoring 25% by detector score is removed. A valid template requires at least 30 retained events and nonzero energy. Residual fitting searches shifts of ±2 samples and shared amplitude scales from 0.5 to 2.0; event timestamps are preserved. Events with insufficient template support retain their GMM assignments.

## Repository layout

| Path | Role |
| --- | --- |
| [`pyproject.toml`](pyproject.toml) | Package metadata, runtime and development dependencies, CLI. |
| [`src/vanillasort/api.py`](src/vanillasort/api.py) | Public `sort()` API, recording access and BaseSorting output. |
| [`src/vanillasort/models/`](src/vanillasort/models/) | Canonical VanillaDet and HuiduRep models, including training methods. |
| [`src/vanillasort/pipeline.py`](src/vanillasort/pipeline.py) / [`ops.py`](src/vanillasort/ops.py) | Published preprocessing, detection, embeddings, amplitude features, GMM and template refinement. |
| [`src/vanillasort/geometry.py`](src/vanillasort/geometry.py) | Local probe neighborhoods and overlap suppression. |
| [`src/vanillasort/checkpoints.py`](src/vanillasort/checkpoints.py) | Model resolution and strict checkpoint loading. |
| [`src/vanillasort/configs/`](src/vanillasort/configs/) | Inference settings and pinned Hugging Face model registry; weights are cached separately. |
| [`src/vanillasort/cli.py`](src/vanillasort/cli.py) / [`run.py`](run.py) | CLI and compatibility launcher. |
| [`tests/`](tests/) | Small synthetic tests using locally generated tiny checkpoints. |
| [`requirements.txt`](requirements.txt) | Historical numerical versions for reproduction comparisons. |

## Citation

**Use of VanillaSort requires citation of the following paper:**

Zishuo Feng and Feng Cao. **Spike Sorting with VanillaSort.** bioRxiv, 2026. [doi:10.64898/2026.09.18.752552](https://www.biorxiv.org/content/10.64898/2026.09.18.752552).

```bibtex
@article{feng2026vanillasort,
  title   = {Spike Sorting with {VanillaSort}},
  author  = {Feng, Zishuo and Cao, Feng},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.09.18.752552},
  url     = {https://www.biorxiv.org/content/10.64898/2026.09.18.752552}
}
```

## License

[GNU Affero General Public License v3.0](LICENSE).
