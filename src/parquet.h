/* Minimal Apache Parquet reader (2026-11-27, atlas: ATLAS-15k ships as Parquet; no Python in the pipeline). Enough of the format for
   flat and nested-list columns written by pyarrow / HF datasets:
   - footer FileMetaData (Thrift compact protocol): schema tree, row groups, column chunks;
   - pages: DATA_PAGE (v1), DATA_PAGE_V2, DICTIONARY_PAGE; codecs UNCOMPRESSED, SNAPPY, GZIP, ZSTD;
   - encodings: PLAIN, PLAIN_DICTIONARY / RLE_DICTIONARY (values), RLE / bit-packed hybrid (levels);
   - physical types BOOLEAN, INT32, INT64, FLOAT, DOUBLE, BYTE_ARRAY (FIXED_LEN_BYTE_ARRAY as bytes).
   Anything else is an error with a message (pq_error), never a silent skip. The file is mmapped read-only.
   A leaf column of one row group decodes to its Dremel triplets: rep[i], def[i] per level entry, and the non-null values packed
   (vals: elem bytes each; BYTE_ARRAY: offs[k] / lens[k] into the chunk's own byte pool). */
#pragma once
#include <stddef.h>
#include <stdint.h>

enum { PQ_BOOLEAN = 0, PQ_INT32 = 1, PQ_INT64 = 2, PQ_INT96 = 3, PQ_FLOAT = 4, PQ_DOUBLE = 5, PQ_BYTE_ARRAY = 6, PQ_FLBA = 7 };

typedef struct {
    char path[256];        /* dotted path, e.g. "x.list.element.list.element" */
    char top[64];          /* the top-level field name ("x") */
    int type, type_len;    /* physical type, FIXED_LEN_BYTE_ARRAY length */
    int max_def, max_rep;
    int rep_def[8];        /* rep_def[r - 1]: the def level at which the repeated node of rep level r is present (r = 1..max_rep) */
} pq_leaf;

typedef struct {
    int codec, type;
    int64_t num_values, data_off, dict_off, comp_size;
} pq_chunkmeta;

typedef struct pq_file pq_file;

typedef struct {
    int64_t n;             /* level entries */
    uint8_t *rep, *def;    /* n each */
    int64_t nv;            /* non-null values */
    void *vals;            /* nv * elem (fixed-width types; BOOLEAN as uint8) */
    int64_t *offs; int32_t *lens; uint8_t *pool;   /* BYTE_ARRAY: value k = pool + offs[k], lens[k] bytes */
} pq_col;

pq_file *pq_open(const char *path);
void pq_close(pq_file *f);
const char *pq_error(void);
int64_t pq_num_rows(const pq_file *f);
int pq_num_row_groups(const pq_file *f);
int64_t pq_rg_rows(const pq_file *f, int rg);
int pq_num_leaves(const pq_file *f);
const pq_leaf *pq_leaf_at(const pq_file *f, int i);
int pq_find_leaf(const pq_file *f, const char *top);   /* the first leaf under top-level field `top`, -1 if none */
const pq_chunkmeta *pq_chunk(const pq_file *f, int rg, int leaf);
const char *pq_created_by(const pq_file *f);
/* decode leaf `leaf` of row group rg into c (zeroed first; pq_col_free after). 0 ok, -1 error (pq_error). */
int pq_read_col(pq_file *f, int rg, int leaf, pq_col *c);
void pq_col_free(pq_col *c);
/* the compression codec name of a Parquet codec id */
const char *pq_codec_name(int codec);
/* snappy raw block decode (exposed for the test): 0 ok */
int pq_snappy_decode(const uint8_t *in, size_t n, uint8_t *out, size_t cap, size_t *outn);
