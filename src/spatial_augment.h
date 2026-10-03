/* Smooth, invertible local CT geometry. Coordinates are ZYX, after the exact
   cube symmetry. Pull: source = R^T (output-center-e(output)) + center.
   Elastic components depend on their own coordinate, so |e'| < .1 makes
   the map invertible; it needs no dense displacement volume. */
#pragma once
#include <math.h>
#include <stdint.h>

typedef struct {
    int active, P;
    double R[3][3], amplitude[3], phase[3], frequency;
} spatial_aug;

void spatial_aug_make(spatial_aug *a,uint64_t seed,int P,float degrees,float rotation_p,float elastic,float elastic_p);
/* Solve the forward map at floating-point sparse supervision coordinates. */
void spatial_aug_forward(const spatial_aug *a,const double source[3],double output[3]);
/* Reusable integer-coordinate sine/derivative tables, 6*P floats. */
void spatial_aug_tables(const spatial_aug *a,float *table);
static inline void spatial_aug_pull(const spatial_aug *a,const float *table,int z,int y,int x,float source[3],float jac[3]) {
    int u[3]={z,y,x}; float c=.5f*(a->P-1),v[3];
    for (int d=0;d<3;d++) { v[d]=u[d]-c-table[d*a->P+u[d]]; jac[d]=1.f/(1.f-table[(3+d)*a->P+u[d]]); }
    for (int d=0;d<3;d++) source[d]=c+(float)a->R[0][d]*v[0]+(float)a->R[1][d]*v[1]+(float)a->R[2][d]*v[2];
}
