/* Read-only zip archive access (zip64 aware) for very large "stored" archives such as the HF labels.zip:
   parses the central directory once, then serves entries by index with pread. */
#pragma once
#include <stddef.h>
#include <stdint.h>

typedef struct zipr zipr;
typedef struct { const char *name; uint64_t size, csize, local_off; int method; } zip_entry;

zipr *zipr_open(const char *path);
void zipr_close(zipr *z);
size_t zipr_count(const zipr *z);
int zipr_entry(const zipr *z, size_t i, zip_entry *e);
/* Decoded entry bytes (stored or deflated); malloc'd, nul-terminated. Thread-safe. */
uint8_t *zipr_read(const zipr *z, size_t i, size_t *len);
/* First entry whose name equals `name`, or -1 (linear scan). */
long zipr_find(const zipr *z, const char *name);
const char *zipr_error(void);
