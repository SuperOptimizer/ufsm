/* Smooth, invertible local CT geometry. Coordinates are ZYX, after the exact
   cube symmetry. Pull: source = scale R^T (output-center-e(output)) + center.
   Elastic components depend on their own coordinate, so |e'| < .1 makes
   the map invertible; it needs no dense displacement volume. */
#pragma once
#include <math.h>
#include <stdint.h>

typedef struct {
    int active, P;
    double R[3][3], amplitude[3], phase[3], frequency;
    double scale;   /* source distance per output voxel (1 = none, < 1 zooms in: the output shows the central P * scale) */
    int affine;     /* anisotropic scale + shear: source - center = scale A R^T v */
    double A[3][3], Ainv[3][3], det;
} spatial_aug;

void spatial_aug_make(spatial_aug *a,uint64_t seed,int P,float degrees,float rotation_p,float elastic,float elastic_p);
/* uniform zoom (scale < 1 zooms in, so every output voxel pulls from inside the source cube) */
void spatial_aug_zoom(spatial_aug *a,double scale);
/* A = diag(exp(e)) (I + shear), e uniform in +/-aniso (log scale), off-diagonal shear uniform in +/-shear. Axes stretched
   past 1 pull from outside the cube there: those voxels are ignored like rotated corners. */
void spatial_aug_affine(spatial_aug *a,uint64_t seed,double aniso,double shear);
/* Solve the forward map at floating-point sparse supervision coordinates. */
void spatial_aug_forward(const spatial_aug *a,const double source[3],double output[3]);
/* Reusable integer-coordinate sine/derivative tables, 6*P floats. */
void spatial_aug_tables(const spatial_aug *a,float *table);
static inline void spatial_aug_pull(const spatial_aug *a,const float *table,int z,int y,int x,float source[3],float jac[3]) {
    int u[3]={z,y,x}; float c=.5f*(a->P-1),v[3];
    for (int d=0;d<3;d++) { v[d]=u[d]-c-table[d*a->P+u[d]]; jac[d]=1.f/(1.f-table[(3+d)*a->P+u[d]]); }
    const float sc=(float)a->scale;
    if (a->affine) {
        float w[3]; for (int d=0;d<3;d++) w[d]=(float)a->R[0][d]*v[0]+(float)a->R[1][d]*v[1]+(float)a->R[2][d]*v[2];
        for (int d=0;d<3;d++) source[d]=c+sc*((float)a->A[d][0]*w[0]+(float)a->A[d][1]*w[1]+(float)a->A[d][2]*w[2]);
        return;
    }
    for (int d=0;d<3;d++) source[d]=c+sc*((float)a->R[0][d]*v[0]+(float)a->R[1][d]*v[1]+(float)a->R[2][d]*v[2]);
}
