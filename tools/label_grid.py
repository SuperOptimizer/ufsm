#!/usr/bin/env python3
"""Native <-> label grid mapping for the scoring tools: the trainer's convention (sources.c mask_nearest).

A label level coarser by f (f = 2 for the 4.8 um labels of a 2.4 um scan) has voxel j centred on native voxel f j: native
voxel g belongs to label voxel (g + f // 2) // f, i.e. label j covers natives f j - f // 2 .. f j - f // 2 + f - 1
({2j - 1, 2j} for f = 2; the rasteriser rounds mesh points with floor(q + 0.5), ingest.c). The older tools pooled natives
{2j, 2j + 1} (`legacy`), half a label voxel off.

Boxes are native [o, o + n) with even o and n; their label grid is [o // 2, o // 2 + n // 2) (the native voxel o + n - 1
falls into the next label voxel and is dropped; label o // 2 also covers native o - 1, outside the box)."""
import numpy as np


def label_index(g, f=2, legacy=False):
    g = np.asarray(g)
    return g // f if legacy else (g + f // 2) // f


def label_box(o, n, f=2):
    """label-grid origin and size of the native box [o, o + n)"""
    o, n = np.asarray(o), np.asarray(n)
    return o // f, n // f


def pool_max(v, o, f=2, legacy=False):
    """native array v (box origin o) -> its label grid, max over each label voxel's natives inside the box"""
    lo, ln = label_box(o, v.shape, f)
    out = v
    for d in range(3):
        idx = label_index(o[d] + np.arange(v.shape[d]), f, legacy) - lo[d]
        keep = idx < ln[d]
        sel = [slice(None)] * 3; sel[d] = keep
        out = out[tuple(sel)]; idx = idx[keep]
        starts = np.r_[0, np.nonzero(np.diff(idx))[0] + 1]
        out = np.maximum.reduceat(out, starts, axis=d)
    return out


def pool_mean(v, o, f=2, legacy=False):
    """as pool_max with the mean (e.g. CT intensity per label voxel)"""
    lo, ln = label_box(o, v.shape, f)
    out = v.astype(np.float32)
    for d in range(3):
        idx = label_index(o[d] + np.arange(v.shape[d]), f, legacy) - lo[d]
        keep = idx < ln[d]
        sel = [slice(None)] * 3; sel[d] = keep
        out = out[tuple(sel)]; idx = idx[keep]
        starts = np.r_[0, np.nonzero(np.diff(idx))[0] + 1]
        cnt = np.diff(np.r_[starts, len(idx)]).astype(np.float32)
        shape = [1, 1, 1]; shape[d] = len(cnt)
        out = np.add.reduceat(out, starts, axis=d) / cnt.reshape(shape)
    return out


def native_centre(j, f=2):
    """native voxel at the centre of label voxel j (absolute indices)"""
    return np.asarray(j) * f


def upsample(lab, o, n, f=2):
    """label-grid array of the box (label_box origin) -> native box by the trainer's nearest mapping"""
    lo, ln = label_box(o, n, f)
    idx = [np.minimum(label_index(o[d] + np.arange(n[d]), f) - lo[d], ln[d] - 1) for d in range(3)]
    return lab[np.ix_(*idx)]
