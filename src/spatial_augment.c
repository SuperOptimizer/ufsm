#include "spatial_augment.h"
#include <string.h>

static uint64_t next(uint64_t *s) { uint64_t z=(*s+=0x9e3779b97f4a7c15ull); z=(z^(z>>30))*0xbf58476d1ce4e5b9ull; z=(z^(z>>27))*0x94d049bb133111ebull; return z^(z>>31); }
static double uniform(uint64_t *s) { return (double)(next(s)>>11)*0x1.0p-53; }
void spatial_aug_make(spatial_aug *a,uint64_t seed,int P,float degrees,float rotation_p,float elastic,float elastic_p) {
    memset(a,0,sizeof *a); a->P=P; a->scale=1; a->det=1;
    for (int d=0;d<3;d++) a->R[d][d]=1;
    if (degrees>0 && uniform(&seed)<rotation_p) {
        a->active=1;
        for (int axis=0;axis<3;axis++) {
            double angle=(2*uniform(&seed)-1)*degrees*(M_PI/180),c=cos(angle),s=sin(angle);
            int i=(axis+1)%3,j=(axis+2)%3;
            for (int k=0;k<3;k++) { double u=a->R[i][k],v=a->R[j][k]; a->R[i][k]=c*u-s*v; a->R[j][k]=s*u+c*v; }
        }
    }
    a->frequency=2*M_PI/fmax(1,P-1);
    if (elastic>0 && uniform(&seed)<elastic_p) {
        a->active=1;
        double bound=fmin(elastic,.09/a->frequency);
        for (int d=0;d<3;d++) { a->amplitude[d]=(2*uniform(&seed)-1)*bound; a->phase[d]=2*M_PI*uniform(&seed); }
    }
}
void spatial_aug_zoom(spatial_aug *a,double scale) { a->scale=scale; if (scale!=1) a->active=1; }
void spatial_aug_affine(spatial_aug *a,uint64_t seed,double aniso,double shear) {
    double e[3],(*A)[3]=a->A,(*I)[3]=a->Ainv;
    for (int d=0;d<3;d++) e[d]=exp((2*uniform(&seed)-1)*aniso);
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) A[i][j]=e[i]*(i==j?1:(2*uniform(&seed)-1)*shear);
    a->det=A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])-A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0])+A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]);
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) {   /* inverse = adjugate / det */
        int r0=(j+1)%3,r1=(j+2)%3,c0=(i+1)%3,c1=(i+2)%3;
        I[i][j]=(A[r0][c0]*A[r1][c1]-A[r0][c1]*A[r1][c0])/a->det;
    }
    a->affine=a->active=1;
}
void spatial_aug_forward(const spatial_aug *a,const double source[3],double output[3]) {
    double c=.5*(a->P-1),q[3];
    if (a->affine) for (int j=0;j<3;j++) q[j]=c+a->Ainv[j][0]*(source[0]-c)+a->Ainv[j][1]*(source[1]-c)+a->Ainv[j][2]*(source[2]-c);
    else for (int j=0;j<3;j++) q[j]=source[j];
    for (int d=0;d<3;d++) {
        double w=c; for (int j=0;j<3;j++) w+=a->R[d][j]*(q[j]-c)/a->scale;
        double u=w;
        /* Newton solves u-e(u)=w. The bounded derivative stays positive. */
        for (int i=0;i<6;i++) { double v=a->frequency*u+a->phase[d]; u-=(u-a->amplitude[d]*sin(v)-w)/(1-a->amplitude[d]*a->frequency*cos(v)); }
        output[d]=u;
    }
}
void spatial_aug_tables(const spatial_aug *a,float *table) {
    for (int d=0;d<3;d++) for (int u=0;u<a->P;u++) {
        double v=a->frequency*u+a->phase[d];
        table[d*a->P+u]=(float)(a->amplitude[d]*sin(v));
        table[(3+d)*a->P+u]=(float)(a->amplitude[d]*a->frequency*cos(v));
    }
}
