/* dino-train GT SupCon points: a reservoir sample of up to cap labelled voxels (codes 1..254) in the window interior [m, P - m)^3,
   optionally only papyrus (raw CT ct > gate; gate < 0: every labelled voxel). Mesh labels include the voids between and inside
   sheets; anchors there teach the embedding the mesh's errors, not the papyrus (upstream dinovol issue; the SSL points were gated
   already). sel[k] = (z P + y) P + x, code[k]; returns the number of candidate voxels seen (the sample holds min(seen, cap)). The
   rng calls are the trainer's (the same sequence as before for gate < 0). */
#pragma once
#include <stdint.h>
#include <stddef.h>
static inline long dn_gt_reservoir(const uint8_t *codes, const uint8_t *ct, int P, int m, float gate, int cap, uint64_t *rs,
                                   uint64_t (*rnext)(uint64_t *), int *sel, int *code) {
    long seen = 0;
    for (int z = m; z < P - m; z++) for (int y = m; y < P - m; y++) {
        const size_t r0 = ((size_t)z * P + y) * P; const uint8_t *row = codes + r0, *crow = ct + r0;
        for (int x = m; x < P - m; x++) {
            const int c = row[x]; if (!c || c == 255) continue;
            if (gate >= 0 && crow[x] <= gate) continue;
            long k = seen++;
            if (k >= cap) { k = (long)(rnext(rs) % (uint64_t)seen); if (k >= cap) continue; }
            sel[k] = (int)(r0 + (size_t)x); code[k] = c;
        }
    }
    return seen;
}
