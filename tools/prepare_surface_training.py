#!/usr/bin/env python3
"""Freeze a completed binary mask level for native training before its pyramid finishes.

Hard links keep the view stable when the builder renames its staging directory.
An exact coarse occupancy array comes from nonempty inner chunks, without
decoding the full mask. It guides cube selection; training uses the finest mask.
"""
import argparse
import ctypes
import ctypes.util
import functools
import json
import os
from pathlib import Path
import shutil
import struct

from build_surface_store import atomic_json


@functools.cache
def crc_table():
    table = []
    for i in range(256):
        c = i
        for _ in range(8):
            c = (c >> 1) ^ (0x82f63b78 if c & 1 else 0)
        table.append(c)
    return table


def crc32c(data):
    table = crc_table()
    crc = 0xffffffff
    for b in data:
        crc = table[(crc ^ b) & 255] ^ (crc >> 8)
    return crc ^ 0xffffffff


def occupancy(array, meta):
    shard = meta['chunk_grid']['configuration']['chunk_shape']
    codec = meta['codecs'][0]
    if codec['name'] != 'sharding_indexed':
        raise ValueError('expected sharded binary masks')
    inner = codec['configuration']['chunk_shape']
    if (inner != [128] * 3 or
            len(set(shard)) != 1 or shard[0] % 128 or
            codec['configuration'].get('index_location') != 'end'):
        raise ValueError('expected ufsm binary-mask shards with an end index')
    cg = shard[0] // 128
    nc = cg**3
    shape = [(n + 127) // 128 for n in meta['shape']]
    data = bytearray(shape[0] * shape[1] * shape[2])
    files = 0
    for path in (array / 'c').glob('*/*/*'):
        if not path.is_file() or path.name.endswith('.tmp'):
            continue
        sz, sy, sx = map(int, path.relative_to(array / 'c').parts)
        with path.open('rb') as f:
            payload = path.stat().st_size - (nc * 16 + 4)
            if payload < 0:
                raise ValueError(f'truncated shard: {path}')
            f.seek(payload)
            index = f.read(nc * 16 + 4)
        if crc32c(index[:-4]) != struct.unpack('<I', index[-4:])[0]:
            raise ValueError(f'bad shard index checksum: {path}')
        for i, (offset, size) in enumerate(struct.iter_unpack('<QQ', index[:-4])):
            if offset == 2**64 - 1 and size == 2**64 - 1:
                continue
            if size <= 0 or offset + size > payload:
                raise ValueError(f'bad chunk extent: {path}')
            z, y, x = sz * cg + i // (cg * cg), sy * cg + (i // cg) % cg, sx * cg + i % cg
            if 0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]:
                data[(z * shape[1] + y) * shape[2] + x] = 255
        files += 1
    if not files or not any(data):
        raise ValueError('completed mask contains no surface chunks')
    return data, shape, files


def write_occupancy(path, data, shape):
    lib = ctypes.CDLL(ctypes.util.find_library('zstd'))
    lib.ZSTD_compressBound.argtypes = [ctypes.c_size_t]
    lib.ZSTD_compressBound.restype = ctypes.c_size_t
    lib.ZSTD_compress.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    lib.ZSTD_compress.restype = ctypes.c_size_t
    lib.ZSTD_isError.argtypes = [ctypes.c_size_t]
    lib.ZSTD_isError.restype = ctypes.c_uint
    chunk = bytearray(128**3)
    encoded = ctypes.create_string_buffer(lib.ZSTD_compressBound(len(chunk)))
    for sz in range((shape[0] + 127) // 128):
        for sy in range((shape[1] + 127) // 128):
            for sx in range((shape[2] + 127) // 128):
                chunk[:] = bytes(len(chunk))
                for z in range(min(128, shape[0] - sz * 128)):
                    for y in range(min(128, shape[1] - sy * 128)):
                        start = ((sz * 128 + z) * shape[1] + sy * 128 + y) * shape[2] + sx * 128
                        count = min(128, shape[2] - sx * 128)
                        dest = (z * 128 + y) * 128
                        chunk[dest:dest + count] = data[start:start + count]
                if not any(chunk):
                    continue
                src = (ctypes.c_ubyte * len(chunk)).from_buffer(chunk)
                n = lib.ZSTD_compress(encoded, len(encoded), src, len(chunk), 3)
                if lib.ZSTD_isError(n):
                    raise RuntimeError('occupancy compression failed')
                file = path / 'c' / str(sz) / str(sy) / str(sx)
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(encoded.raw[:n])
    atomic_json(path / 'zarr.json', {'zarr_format': 3, 'node_type': 'array', 'shape': shape, 'data_type': 'uint8',
        'chunk_grid': {'name': 'regular', 'configuration': {'chunk_shape': [128]*3}},
        'chunk_key_encoding': {'name': 'default', 'configuration': {'separator': '/'}}, 'fill_value': 0,
        'codecs': [{'name': 'bytes', 'configuration': {'endian': 'little'}},
                   {'name': 'zstd', 'configuration': {'level': 3}}],
        'attributes': {'ufsm': {'encoding': 'binary', 'content': 'sampling occupancy', 'pool': 'any-positive-128'}}})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--sources-out', type=Path, required=True)
    p.add_argument('--source-template', type=Path, required=True)
    a = p.parse_args()
    cfg = json.loads(a.source_template.read_text())
    if len(cfg['sources']) != 1:
        raise ValueError('expected one merged source')
    build = json.loads((a.work / 'build.json').read_text())
    if build.get('encoding') != 'binary' or not 0 <= build['label_level'] <= 2:
        raise ValueError('expected a binary build whose occupancy fits the supported CT rungs')
    source = cfg['sources'][0]
    if source['ct'] != build['ct'] or source['um'] != build['native_um']:
        raise ValueError('source template belongs to another CT or native voxel size')
    if build['status'] != 'complete' and f"level {build['label_level']} rasterized in " not in (a.work / 'raster.log').read_text():
        raise ValueError('finest mask level is not complete')
    root = Path(build['store']) if build['status'] == 'complete' else Path(build['command'][2])
    fine_um = build['label_um']
    array = root / f'{fine_um:.10g}'
    meta = json.loads((array / 'zarr.json').read_text())
    if meta['fill_value'] != 0 or meta['attributes']['ufsm']['encoding'] != 'binary':
        raise ValueError('expected zero-filled binary masks')
    data, shape, files = occupancy(array, meta)
    out = a.out.resolve()
    staging = out.with_name(out.name + '.building')
    if out.exists() or staging.exists():
        raise FileExistsError(out)
    staging.mkdir(parents=True)
    shutil.copytree(array, staging / array.name, copy_function=os.link)
    coarse_um = fine_um * 128
    write_occupancy(staging / f'{coarse_um:.10g}', data, shape)
    datasets = [{'path': f'{um:.10g}', 'coordinateTransformations': [{'type': 'scale', 'scale': [um]*3}]} for um in [fine_um, coarse_um]]
    atomic_json(staging / 'zarr.json', {'zarr_format': 3, 'node_type': 'group', 'attributes': {'ome': {'version': '0.5',
        'multiscales': [{'version': '0.5', 'name': 'native surface training',
                        'axes': [{'name': d, 'type': 'space', 'unit': 'micrometer'} for d in ['z','y','x']], 'datasets': datasets}]}}})
    atomic_json(staging / 'provenance.json', dict(build, training_view=True, original_array=str(array),
                source_shards=files, occupied_cells=data.count(255), occupancy_um=coarse_um,
                training_levels=[0], note='Fine mask immutable hard links; coarse array only guides native sampling.'))
    staging.rename(out)
    cfg['sources'][0]['targets']['recto'].update(root=str(out), encoding='binary', min_level=build['label_level'])
    atomic_json(a.sources_out, cfg)
    print(f"TRAINING_MASK_READY {out}: {files} shards, {data.count(255)} occupied cells", flush=True)


if __name__ == '__main__':
    main()
