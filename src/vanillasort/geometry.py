"""Map recording geometry to the checkpoint's four physical channel slots."""

from __future__ import annotations

import numpy as np


def local_neighborhoods(recording, mode="dense"):
    """Return nearest-four patches and their uniquely owned channel indices.

    Groups and probe/shank boundaries are respected. Channels inside a patch
    retain recording order because the published convolution is order-sensitive.
    The detector has no missing-channel mask: fewer than four real contacts in
    a group is an error, rather than an unsupported artificial electrode.

    "cover" greedily chooses the patch covering the most still-uncovered
    contacts (stable ties in recording order). Every contact occurs in a patch
    and owns exactly one selected patch; it need not itself be a patch center.
    "dense" retains the original unique nearest-four patch for every center.
    """
    if mode not in ("cover", "dense"):
        raise ValueError("neighborhood_mode must be 'cover' or 'dense'")
    if recording.has_probe() and recording.is_probe_3d():
        raise ValueError("The published checkpoints require 2D geometry; select an explicit 2D projection first")
    locations = np.asarray(recording.get_channel_locations(), dtype=np.float32)
    count = recording.get_num_channels()
    if locations.shape != (count, 2) or not np.isfinite(locations).all():
        raise ValueError("VanillaSort requires finite 2D channel locations in micrometres")
    groups = recording.get_property("group")
    if groups is None:
        groups = np.zeros(count, dtype=int)
    contacts = recording.get_property("contact_vector")
    partitions = {}
    for index in range(count):
        key = (groups[index].item(),)
        if contacts is not None:
            for name in ("probe_index", "shank_ids"):
                if name in contacts.dtype.names:
                    key += (contacts[name][index].item(),)
        partitions.setdefault(key, []).append(index)
    patches = {}
    for indices in partitions.values():
        indices = np.asarray(indices, dtype=np.int64)
        if len(indices) < 4:
            raise ValueError(
                "Each channel group/shank needs at least four real channels; VanillaDet has no channel mask"
            )
        if len(np.unique(locations[indices], axis=0)) != len(indices):
            raise ValueError("Channel locations must be unique within each group/shank")
        for center in indices:
            distance = np.sum((locations[indices] - locations[center]) ** 2, axis=1)
            nearest = indices[np.argsort(distance, kind="stable")[:4]]
            patch = tuple(sorted(nearest.tolist()))
            patches.setdefault(patch, []).append(int(center))
    if mode == "dense":
        return locations, [(np.asarray(patch), np.asarray(owners)) for patch, owners in patches.items()]
    # Work on candidate channel sets, not one detector pass per center. Use
    # newly covered contacts as owners, so overlapping patches do not both
    # claim the same primary channel and no covered contact loses ownership.
    candidates = list(patches)
    channel_sets = [set(patch) for patch in candidates]
    uncovered = set(range(count))
    selected = []
    while uncovered:
        best = max(range(len(candidates)), key=lambda i: len(channel_sets[i] & uncovered))
        owners = channel_sets[best] & uncovered
        if not owners:
            raise RuntimeError("Nearest-four patches failed to cover all recording channels")
        selected.append((np.asarray(candidates[best], dtype=np.int64), np.asarray(sorted(owners), dtype=np.int64)))
        uncovered.difference_update(owners)
    return locations, selected


def deduplicate_events(samples, scores, patch_indices, patches, radius):
    """Score-priority temporal NMS only between intersecting spatial patches.

    Each patch has already applied the published NMS. Spatially disjoint groups
    may therefore retain simultaneous spikes. Stable sorting resolves ties.
    """
    keep = np.zeros(len(samples), dtype=bool)
    blocked = np.zeros(len(samples), dtype=bool)
    chronological = np.argsort(samples, kind="stable")
    ordered_times = samples[chronological]
    patch_channels = [set(channels.tolist()) for channels, _ in patches]
    overlaps = np.array([[bool(a & b) for b in patch_channels] for a in patch_channels])
    for event in np.argsort(-scores, kind="stable"):
        if blocked[event]:
            continue
        keep[event] = True
        sample = int(samples[event])
        first = np.searchsorted(ordered_times, sample - radius, side="left")
        last = np.searchsorted(ordered_times, sample + radius, side="right")
        neighbors = chronological[first:last]
        neighbors = neighbors[overlaps[patch_indices[event], patch_indices[neighbors]]]
        blocked[neighbors] = True
    return keep
