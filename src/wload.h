/* Weight files of external PyTorch models, read in C (no Python conversion step): safetensors (JSON header + raw tensors) and
   PyTorch zip checkpoints (torch.save: <dir>/data.pkl + <dir>/data/<key> storages), the latter through a minimal pickle VM that
   knows the tensor-rebuild opcodes (_rebuild_tensor_v2 / _rebuild_parameter over persistent-id storages, OrderedDict, plain
   containers and scalars). Every tensor is converted to a contiguous float32 array (f32 / f16 / bf16 / f64 / integer storages).

   A .pth holds a dict; the state dict is the entry named `sub` (e.g. "ema_model", "model"), or with sub nullptr the top-level dict
   when it holds tensors, else the first of "ema_model", "model", "state_dict" or any other entry that is a dict of tensors.
   The top-level numbers and strings of a .pth (e.g. sampling_um, architecture_family) stay readable (wl_num / wl_str). */
#pragma once
#include <stddef.h>
#include <stdint.h>

typedef struct { char *name; int ndim; int64_t shape[8]; int64_t numel; float *data; } wl_tensor;
typedef struct wl_dict wl_dict;

wl_dict *wl_load(const char *path, const char *sub);   /* nullptr on failure (wl_error) */
void wl_free(wl_dict *d);
int wl_count(const wl_dict *d);
const wl_tensor *wl_at(const wl_dict *d, int i);
const wl_tensor *wl_get(const wl_dict *d, const char *name);            /* nullptr if absent */
const wl_tensor *wl_getf(const wl_dict *d, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
double wl_num(const wl_dict *d, const char *key, double dflt);          /* top-level number of a .pth (bools 0/1) */
const char *wl_str(const wl_dict *d, const char *key);                  /* top-level string of a .pth, nullptr if absent */
const char *wl_error(void);
