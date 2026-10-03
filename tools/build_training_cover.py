#!/usr/bin/env python3
"""Build an immutable shuffled, once-only native-volume training cover.

Tiles cover the occupied-cell bounding box, with held-out boxes removed. A tile
is retained if it intersects a positive occupancy cell. This is a conservative
cover: some retained tiles may contain no fine positives, and clamped tail tiles
overlap. Every planned tile is visited once; voxels need not be visited once.
Origins must not subsequently be snapped or position-jittered by the sampler.
Only CPU metadata reads and the ufsm CPU volume reader are used.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import subprocess
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_metadata(root, key):
    if '://' in root:
        location = root.rstrip('/') + '/' + key.strip('/') + '/zarr.json'
        with urllib.request.urlopen(location, timeout=60) as response:
            raw = response.read()
    else:
        location = Path(root) / key / 'zarr.json'
        raw = location.read_bytes()
    return json.loads(raw), digest(raw)


def datasets(meta):
    attrs = meta.get('attributes', {})
    scales = attrs.get('ome', attrs).get('multiscales', [])
    if not scales:
        raise ValueError('pyramid metadata has no multiscale datasets')
    result = []
    for item in scales[0]['datasets']:
        transforms = item.get('coordinateTransformations', [])
        scale = next((t['scale'] for t in transforms if t['type'] == 'scale'), None)
        if not scale or len(scale) != 3 or any(v <= 0 for v in scale):
            raise ValueError('expected positive three-dimensional pyramid scales')
        result.append((item['path'], scale))
    return result


def shape3(meta):
    shape = meta.get('shape', [])
    if len(shape) != 3 or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError('expected a positive three-dimensional array shape')
    return tuple(shape)


def local_root(root):
    return root if '://' in root else str(Path(root).expanduser().resolve())


class Occupancy:
    """Compact per-(z,y) bitsets; an intersecting-cell query needs few rows."""

    def __init__(self, data, shape, cell, native_shape):
        self.shape, self.cell, self.native_shape = tuple(shape), tuple(cell), tuple(native_shape)
        if len(data) != math.prod(shape):
            raise ValueError('occupancy byte count does not match its metadata shape')
        self.sha256 = digest(data)
        self.rows = []
        bounds_lo, bounds_hi = list(shape), [-1] * 3
        self.count = 0
        nz, ny, nx = shape
        for z in range(nz):
            for y in range(ny):
                offset = (z * ny + y) * nx
                row = 0
                for x, value in enumerate(data[offset:offset + nx]):
                    if value:
                        row |= 1 << x
                        self.count += 1
                        for d, v in enumerate((z, y, x)):
                            bounds_lo[d] = min(bounds_lo[d], v)
                            bounds_hi[d] = max(bounds_hi[d], v)
                self.rows.append(row)
        if not self.count:
            raise ValueError('occupancy contains no positive cells')
        self.bounds = tuple(bounds_lo[d] * cell[d] for d in range(3)) + tuple(
            min((bounds_hi[d] + 1) * cell[d], native_shape[d]) for d in range(3))

    def intersects(self, lo, hi):
        start = [max(0, lo[d] // self.cell[d]) for d in range(3)]
        end = [min(self.shape[d], (hi[d] + self.cell[d] - 1) // self.cell[d]) for d in range(3)]
        if any(start[d] >= end[d] for d in range(3)):
            return False
        bits = ((1 << (end[2] - start[2])) - 1) << start[2]
        ny = self.shape[1]
        return any(self.rows[z * ny + y] & bits
                   for z in range(start[0], end[0]) for y in range(start[1], end[1]))


def intersects(a, b):
    return all(a[d] < b[d + 3] and b[d] < a[d + 3] for d in range(3))


def subtract_box(box, hole):
    """Disjoint rectangular partition of box minus hole, at most six pieces."""
    if not intersects(box, hole):
        return [tuple(box)]
    middle = list(box)
    result = []
    for d in range(3):
        start, end = max(middle[d], hole[d]), min(middle[d + 3], hole[d + 3])
        if middle[d] < start:
            piece = middle.copy(); piece[d + 3] = start
            result.append(tuple(piece))
        if end < middle[d + 3]:
            piece = middle.copy(); piece[d] = end
            result.append(tuple(piece))
        middle[d], middle[d + 3] = start, end
    return result


def axis_origins(lo, hi, P):
    if hi - lo < P:
        raise ValueError('an occupied holdout boundary strip is narrower than P')
    result = list(range(lo, hi - P + 1, P))
    if result[-1] + P < hi:
        result.append(hi - P)
    return result


def core_axes(core, bounds, P):
    """Expand narrow middle dimensions for context, then verify holdout safety."""
    axes = []
    for d in range(3):
        lo, hi = core[d], core[d + 3]
        if hi - lo < P:
            lo = max(bounds[d], min((lo + hi - P) // 2, bounds[d + 3] - P))
            hi = lo + P
        axes.append(axis_origins(lo, hi, P))
    return axes


def planned_tiles(occ, P, holdouts):
    bounds = occ.bounds
    if any(bounds[d + 3] - bounds[d] < P for d in range(3)):
        raise ValueError('occupied bounding box is narrower than P')
    cores = [bounds]
    for hole in holdouts:
        cores = [piece for box in cores for piece in subtract_box(box, hole)]
    tiles, certificates = set(), []
    for core in cores:
        if not occ.intersects(core[:3], core[3:]):
            continue
        axes = core_axes(core, bounds, P)
        # Each interval starts before the core, ends after it, and consecutive
        # starts differ by <= P. Their Cartesian product therefore covers core.
        for d, values in enumerate(axes):
            if (values[0] > core[d] or values[-1] + P < core[d + 3] or
                    any(b - a > P for a, b in zip(values, values[1:]))):
                raise ValueError('tile intervals leave a coverage gap')
        candidates, retained = 0, 0
        for origin in itertools.product(*axes):
            end = tuple(v + P for v in origin)
            if not occ.intersects(origin, end):
                continue
            tile = origin + end
            if any(intersects(tile, h) for h in holdouts):
                raise ValueError('holdout leaves an occupied strip that cannot fit a safe P cube')
            if any(origin[d] < bounds[d] or end[d] > bounds[d + 3] for d in range(3)):
                raise ValueError('tile escapes the occupied bounding box')
            candidates += 1
            if origin not in tiles:
                tiles.add(origin); retained += 1
        certificates.append({'core': list(core), 'axis_origins': axes,
                             'retained_tiles': retained, 'occupied_candidates': candidates})
    if not tiles:
        raise ValueError('no occupied training tiles remain after holdout exclusion')
    return tiles, certificates


def verify_plan(plan, occ, holdouts):
    """Verify the interval coverage proof and exact immutable tile membership.

    A point in an occupied cell outside the holdouts lies in one partition core.
    The Cartesian interval grid covers that core. The tile containing the point
    intersects its positive coarse cell, so the occupancy filter retains it.
    No assumption about where positives lie inside a coarse cell is necessary.
    """
    if plan['version'] != 1 or plan['level'] != 0 or plan['bounds'] != list(occ.bounds):
        raise ValueError('invalid plan version, native level or bounds')
    rows = plan['tiles']
    if plan['count'] != len(rows) or not rows:
        raise ValueError('plan count does not match its tiles')
    if any(len(row) != 4 or row[0] != 0 or any(type(v) is not int for v in row) for row in rows):
        raise ValueError('invalid source or tile coordinates')
    actual = {tuple(row[1:]) for row in rows}
    if len(actual) != len(rows):
        raise ValueError('plan contains duplicate tiles')
    expected, _ = planned_tiles(occ, plan['P'], holdouts)
    if actual != expected:
        raise ValueError('plan tile membership does not provide the declared complete cover')


def make_plan(occ, P, seed, source_name, holdouts, binding):
    tiles, certificates = planned_tiles(occ, P, holdouts)
    rows = [[0, *origin] for origin in sorted(tiles)]
    random.Random(seed).shuffle(rows)
    plan = {'version': 1, 'P': P, 'level': 0, 'seed': seed, 'count': len(rows),
            'source_names': [source_name], 'bounds': list(occ.bounds), 'tiles': rows,
            'binding': binding, 'coverage': {
                'basis': 'Every occupied coarse-cell box outside holdouts is covered; each tile is used once.',
                'ordering': 'immutable seeded shuffle', 'position_jitter': False, 'snap': False,
                'boundary_tiles_overlap': True, 'occupancy_cells': occ.count,
                'holdouts': [list(h) for h in holdouts], 'slabs': certificates}}
    verify_plan(plan, occ, holdouts)
    return plan


def build(sources_path, P, seed, occupancy_raw=None, binary=ROOT / 'build/ufsm'):
    if type(P) is not int or P <= 0:
        raise ValueError('P must be positive')
    raw_sources = Path(sources_path).read_bytes()
    cfg = json.loads(raw_sources)
    if len(cfg.get('sources', [])) != 1:
        raise ValueError('coverage currently requires one merged source')
    source = cfg['sources'][0]
    if not source.get('name') or not math.isfinite(source['um']) or source['um'] <= 0:
        raise ValueError('invalid source name or native voxel size')
    target = source['targets']['recto']
    if not isinstance(target, dict) or target.get('encoding') != 'binary':
        raise ValueError('coverage requires a merged binary pyramid target')
    ct_root, label_root = local_root(source['root']), local_root(target.get('root', source['root']))
    ct_key, label_group = source['ct'].strip('/'), target.get('group', '.').strip('/')
    ct_group, ct_group_sha = read_metadata(ct_root, ct_key)
    ct_levels = datasets(ct_group)
    # ufsm accepts both physical-micrometer OME scales and dimensionless 1,2,4
    # scales, which are used by the upstream Paris4 CT metadata.
    relative = all(math.isclose(v, 1.0, rel_tol=1e-8) for v in ct_levels[0][1])
    native_scale = 1.0 if relative else source['um']
    ct_array = next((key for key, scale in ct_levels
                     if all(math.isclose(v, native_scale, rel_tol=1e-8) for v in scale)), None)
    if ct_array is None:
        raise ValueError('CT has no array at the configured native resolution')
    ct_meta, ct_meta_sha = read_metadata(ct_root, ct_key + '/' + ct_array)
    native_shape = shape3(ct_meta)
    group_meta, label_meta_sha = read_metadata(label_root, label_group)
    if '://' in label_root:
        with urllib.request.urlopen(label_root.rstrip('/') + '/' + label_group + '/provenance.json', timeout=60) as response:
            provenance_raw = response.read()
    else:
        provenance_raw = (Path(label_root) / label_group / 'provenance.json').read_bytes()
    provenance = json.loads(provenance_raw)
    if (provenance.get('native_shape_zyx') != list(native_shape) or
            provenance.get('ct') != source['ct'] or
            not math.isclose(provenance.get('native_um', 0), source['um'], rel_tol=1e-8)):
        raise ValueError('label provenance belongs to another CT shape or resolution')
    coarse = None
    for key, scale in sorted(datasets(group_meta), key=lambda item: math.prod(item[1]), reverse=True):
        meta, meta_sha = read_metadata(label_root, label_group + '/' + key)
        attrs = meta.get('attributes', {}).get('ufsm', {})
        if attrs.get('content') == 'sampling occupancy':
            coarse = key, scale, meta, meta_sha
            break
    if coarse is None:
        raise ValueError('label training view has no sampling occupancy array')
    key, scale, meta, occupancy_meta_sha = coarse
    cell = tuple(round(v / source['um']) for v in scale)
    if any(n <= 0 or not math.isclose(v / source['um'], n, rel_tol=1e-8) for n, v in zip(cell, scale)):
        raise ValueError('occupancy cells must be integer multiples of native voxels')
    occupancy_shape = shape3(meta)
    if occupancy_shape != tuple((n + f - 1) // f for n, f in zip(native_shape, cell)):
        raise ValueError('occupancy shape does not cover the native CT grid')
    if occupancy_raw:
        data = Path(occupancy_raw).read_bytes()
    else:
        with tempfile.TemporaryDirectory(prefix='ufsm-cover-') as tmp:
            path = Path(tmp) / 'occupancy.raw'
            cmd = [str(binary), 'read', label_root, label_group + '/' + key,
                   '0', '0', '0', *map(str, occupancy_shape), str(path), '--threads', '4']
            subprocess.run(cmd, check=True)
            data = path.read_bytes()
    occ = Occupancy(data, occupancy_shape, cell, native_shape)
    h = source.get('holdout')
    if h and (len(h) != 6 or any(type(v) is not int or v < 0 for v in h[:3]) or
              any(type(v) is not int or v <= 0 for v in h[3:])):
        raise ValueError('invalid source holdout box')
    holdouts = [tuple(h[:3]) + tuple(h[d] + h[d + 3] for d in range(3))] if h else []
    binding = {'source_config_sha256': digest(raw_sources), 'ct_root': ct_root,
               'ct': source['ct'], 'native_um': source['um'], 'native_shape_zyx': list(native_shape),
               'ct_group_metadata_sha256': ct_group_sha, 'ct_array_metadata_sha256': ct_meta_sha,
               'label_root': label_root, 'label_group': label_group,
               'label_group_metadata_sha256': label_meta_sha,
               'label_provenance_sha256': digest(provenance_raw),
               'occupancy_array': key, 'occupancy_shape_zyx': list(occupancy_shape),
               'occupancy_native_cell_zyx': list(cell), 'occupancy_metadata_sha256': occupancy_meta_sha,
               'occupancy_sha256': occ.sha256}
    return make_plan(occ, P, seed, source['name'], holdouts, binding)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--P', type=int, default=704)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--occupancy-raw', type=Path, help='predecoded uint8 occupancy; metadata still validated')
    parser.add_argument('--binary', type=Path, default=ROOT / 'build/ufsm')
    args = parser.parse_args(argv)
    plan = build(args.sources, args.P, args.seed, args.occupancy_raw, args.binary)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + '.tmp')
    temporary.write_text(json.dumps(plan, separators=(',', ':')) + '\n')
    temporary.replace(args.out)
    print(f"TRAINING_COVER_READY {args.out}: {plan['count']} distinct {args.P}^3 tiles, "
          f"seed {args.seed}, sha256 {digest(args.out.read_bytes())}", flush=True)


if __name__ == '__main__':
    main()
