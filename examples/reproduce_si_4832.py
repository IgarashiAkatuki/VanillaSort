"""Reproduce the 64-channel, 60-second recording from SpikeInterface PR #4832.

Install both local checkouts before running this script. Each output folder
must be new; the same recording cache can be reused for before/after runs.
"""

import argparse
from pathlib import Path
import time

import spikeinterface.full as si


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New SpikeInterface sorter output folder")
    parser.add_argument("--cache", type=Path, default=Path("output/si4832-rec-cache"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-path", type=Path, help="Optional local copy of the published checkpoints")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output folder; previous results are retained")
    if args.cache.exists():
        recording = si.load(args.cache)
    else:
        # Fix the input seed to compare versions using exactly the same traces.
        recording, _ = si.generate_ground_truth_recording(
            num_channels=64, durations=[60], sampling_frequency=30_000, seed=0
        )
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        recording = recording.save(folder=args.cache, n_jobs=-1)
    if (
        recording.get_num_channels() != 64
        or recording.get_num_segments() != 1
        or recording.get_num_samples() != 1_800_000
        or recording.get_sampling_frequency() != 30_000
    ):
        parser.error("The cache must contain one 60-second, 64-channel, 30 kHz recording")
    started = time.perf_counter()
    sorting = si.run_sorter(
        "vanillasort",
        recording,
        folder=args.output,
        verbose=True,
        device=args.device,
        model_path=args.model_path,
    )
    print(
        f"Finished in {time.perf_counter() - started:.1f}s: "
        f"{sorting.get_num_units()} units, {len(sorting.to_spike_vector())} spikes",
        flush=True,
    )


if __name__ == "__main__":
    main()
