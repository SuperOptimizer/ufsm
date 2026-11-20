/* src/occcache.h: the sampler's draw-cell index (source_occupancy) cached on disk for sources whose labels are computed on the fly from
   fibre polylines (2026-11-20). Their scan stamps every fibre of the set at the coarse level: Paris 4's 2.47e9 level-5 cells took
   ~15-20 min of one core per trainer launch on the laptop. The index lives beside the derived-label cache (dcache_key_dir under
   $UFSM_DERIVED_CACHE / ~/.cache/ufsm/derived) under a key naming everything it depends on: the source's CT and levels, iso, every
   fibre spec (files with size and mtime, radius / guard / ring / iso / piso / plane) and a format version. Deleting it is safe. */
#pragma once
#include "sources.h"
#include "sample.h"
#include <stddef.h>

/* the key of s's occupancy; 0 (and the key in k) when s is cacheable (a source with an on-the-fly fibre target), else -1 */
int occ_cache_key(const source *s, char *k, size_t nk);
/* load: 0 and *o filled (idx malloc'd) on a hit, -1 on a miss; save: 0 ok. dir: the cache root (nullptr / "" = none: a miss / no-op) */
int occ_cache_load(const char *dir, const char *key, occ_index *o);
int occ_cache_save(const char *dir, const char *key, const occ_index *o);
