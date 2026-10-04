/* Experimental surface confidence + continuous winding coordinate task.
   Geometry files are little-endian native-voxel ZYX records written by
   tools/build_sheet_geometry.py. No dense floating-point teachers. */
#pragma once
#include <stddef.h>
#include <stdint.h>

#define SHEET_PATH 32
typedef struct { uint32_t kind, count; float weight, target; float points[SHEET_PATH][4]; } sheet_record;
typedef struct { int32_t cell[3]; uint32_t count; uint64_t offset; } sheet_index;
typedef struct { float xyz[3], q, q0; } sheet_point;
typedef struct { uint32_t kind, count, first; float weight, target; } sheet_term;
typedef struct {
    size_t np, nt;
    sheet_point *points;
    sheet_term *terms;
} sheet_batch;
typedef struct {
    int cell_size;
    size_t nr, ni, records_bytes, index_bytes;
    const sheet_record *records;
    const sheet_index *index;
    int nk;
    double (*knots)[5];
    double center, scale;
    float (*contacts)[4]; size_t nc;
    double max_soft_sigma;
    char manifest_sha[65], reference_sha[65];
} sheet_dataset;

sheet_dataset *sheet_load(const char *manifest);
sheet_dataset *sheet_load_reference(const char *path);
void sheet_free(sheet_dataset *s);
double sheet_reference(const sheet_dataset *s, const double xyz[3]);
void sheet_parameters(const sheet_dataset *s,double z,double params[4]);
/* Widen the audited cap with matching exclusions around close ordering pairs.
   Apply once after loading; records/reference and their hashes stay immutable. */
int sheet_widen_bands(sheet_dataset *s,double scale);
/* Mark unresolved close contacts in the unaugmented native-voxel ignore mask. */
void sheet_mask_contacts(const sheet_dataset *s,const int64_t origin[3],int P,uint8_t *ignore);
/* Points transformed to augmented local cube coordinates. Empty batches valid. */
sheet_batch *sheet_sample(const sheet_dataset *s, const int64_t origin[3], int P,
                         const int perm[3], const int flip[3], const uint8_t *ct,
                         const uint8_t *surface_target,const uint8_t *loss_mask,uint64_t seed);
sheet_batch *sheet_clone(const sheet_batch *b);
void sheet_batch_free(sheet_batch *b);
/* values = [point][surface logit, complete winding q], FP64 reference reduction.
   Returns weighted loss, writes dloss/dsampled logits and five unweighted terms. */
double sheet_loss(const sheet_batch *b, const double *values, double *grad, double parts[5], float ramp, int variant);
/* Checkpoint task contract; 0 legacy, 1 winding, -1 invalid. */
int sheet_checkpoint(const char *path, char manifest_sha[65], char reference_sha[65]);
int sheet_checkpoint_options(const char *path,int *variant,int *schedule_start);
