#include "sheet.h"
#include "json.h"
#include "checkpoint.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <openssl/evp.h>

_Static_assert(sizeof(sheet_record) == 528, "sheet record layout");
_Static_assert(sizeof(sheet_index) == 24, "sheet index layout");

static char *read_file(const char *path, size_t *n) {
    FILE *f=fopen(path,"rb"); if (!f) return nullptr;
    if (fseek(f,0,SEEK_END)) { fclose(f); return nullptr; }
    long size=ftell(f); rewind(f); if (size<0 || size > 64*1024*1024) { fclose(f); return nullptr; }
    char *s=malloc((size_t)size+1); if (!s) { fclose(f); return nullptr; }
    *n=fread(s,1,(size_t)size,f); fclose(f);
    if (*n!=(size_t)size) { free(s); return nullptr; } s[*n]=0; return s;
}
static int hash_file(const char *path,char out[65]) {
    FILE *f=fopen(path,"rb"); if (!f) return -1;
    EVP_MD_CTX *ctx=EVP_MD_CTX_new(); EVP_DigestInit_ex(ctx,EVP_sha256(),nullptr);
    unsigned char buffer[65536],hash[32]; size_t n; unsigned hn;
    while ((n=fread(buffer,1,sizeof buffer,f))) EVP_DigestUpdate(ctx,buffer,n);
    int bad=ferror(f); fclose(f); EVP_DigestFinal_ex(ctx,hash,&hn); EVP_MD_CTX_free(ctx);
    for (int i=0;i<32;i++) sprintf(out+2*i,"%02x",hash[i]);
    out[64]=0; return bad?-1:0;
}
static void *map_file(const char *path,size_t *bytes) {
    int fd=open(path,O_RDONLY); if (fd<0) return nullptr; struct stat st;
    if (fstat(fd,&st) || st.st_size<=0) { close(fd); return nullptr; }
    *bytes=(size_t)st.st_size; void *p=mmap(nullptr,*bytes,PROT_READ,MAP_PRIVATE,fd,0); close(fd);
    return p==MAP_FAILED?nullptr:p;
}
sheet_dataset *sheet_load_reference(const char *path) {
    size_t n; char *text=read_file(path,&n); if (!text) return nullptr;
    json *r=json_parse(text,n); free(text); if (!r) return nullptr;
    sheet_dataset *s=calloc(1,sizeof *s); const json *knots=json_get(r,"knots");
    s->center=json_num(json_get(r,"input_center"),NAN); s->scale=json_num(json_get(r,"input_scale"),NAN);
    if (json_num(json_get(r,"version"),0)!=1 || strcmp(json_str(json_get(r,"units"),""),"turns") || strcmp(json_str(json_get(r,"coordinate_order"),""),"zyx") ||
        !knots || knots->type!=J_ARR || knots->n<2 || !isfinite(s->center) || !isfinite(s->scale) || s->scale<=0 || hash_file(path,s->reference_sha)) goto bad_ref;
    s->nk=(int)knots->n; s->knots=malloc(knots->n*sizeof *s->knots);
    for (int i=0;i<s->nk;i++) for (int d=0;d<5;d++) {
        s->knots[i][d]=json_num(json_at(json_at(knots,i),d),NAN);
        if (!isfinite(s->knots[i][d]) || (d==3 && s->knots[i][d]<=0) || (d==0 && i && s->knots[i][d]<=s->knots[i-1][d])) goto bad_ref;
    }
    json_free(r); return s;
bad_ref:
    json_free(r); sheet_free(s); return nullptr;
}
sheet_dataset *sheet_load(const char *manifest) {
    uint16_t endian=1; if (*(uint8_t *)&endian!=1) return nullptr;
    size_t n; char *text=read_file(manifest,&n); if (!text) return nullptr;
    json *j=json_parse(text,n); free(text); if (!j) return nullptr;
    sheet_dataset *s=calloc(1,sizeof *s); char dir[4096],path[4300],sha[65];
    snprintf(dir,sizeof dir,"%s",manifest); char *slash=strrchr(dir,'/'); if (slash) *slash=0; else strcpy(dir,".");
    if (json_num(json_get(j,"version"),0)!=1 || strcmp(json_str(json_get(j,"task"),""),"surface_winding") ||
        json_num(json_get(j,"record_bytes"),0)!=sizeof(sheet_record) || json_num(json_get(j,"index_bytes"),0)!=sizeof(sheet_index)) goto bad;
    s->cell_size=(int)json_num(json_get(j,"cell_size"),0); if (s->cell_size<1 || hash_file(manifest,s->manifest_sha)) goto bad;
    const char *names[]={"records.bin","index.bin","reference.json","audit.json","contacts.bin"};
    for (int k=0;k<5;k++) {
        snprintf(path,sizeof path,"%s/%s",dir,names[k]);
        if (hash_file(path,sha) || strcmp(sha,json_str(json_get(json_get(j,"files"),names[k]),""))) goto bad;
        if (k==0) s->records=map_file(path,&s->records_bytes);
        if (k==1) s->index=map_file(path,&s->index_bytes);
        if (k==4) {
            struct stat st; if (stat(path,&st) || st.st_size%16) goto bad;
            s->nc=(size_t)st.st_size/16;
            if (s->nc!=json_num(json_get(j,"contacts"),-1)) goto bad;
            s->contacts=malloc(st.st_size?st.st_size:1);
            FILE *f=fopen(path,"rb"); if (!f) goto bad;
            int read_bad=fread(s->contacts,16,s->nc,f)!=s->nc; fclose(f); if (read_bad) goto bad;
            for (size_t c=0;c<s->nc;c++) {
                for (int d=0;d<4;d++) if (!isfinite(s->contacts[c][d])) goto bad;
                if (s->contacts[c][3]<=0 || s->contacts[c][3]>32 || (c && s->contacts[c][0]<s->contacts[c-1][0])) goto bad;
            }
        }
        if (k==2) {
            strcpy(s->reference_sha,sha); text=read_file(path,&n); if (!text) goto bad;
            json *r=json_parse(text,n); free(text); if (!r) goto bad;
            const json *knots=json_get(r,"knots");
            s->center=json_num(json_get(r,"input_center"),NAN); s->scale=json_num(json_get(r,"input_scale"),NAN);
            if (json_num(json_get(r,"version"),0)!=1 || strcmp(json_str(json_get(r,"units"),""),"turns") || strcmp(json_str(json_get(r,"coordinate_order"),""),"zyx") ||
                !knots || knots->type!=J_ARR || knots->n<2 || !isfinite(s->center) || !isfinite(s->scale) || s->scale<=0) { json_free(r); goto bad; }
            s->nk=(int)knots->n; s->knots=malloc(knots->n*sizeof *s->knots);
            for (int i=0;i<s->nk;i++) for (int d=0;d<5;d++) {
                s->knots[i][d]=json_num(json_at(json_at(knots,i),d),NAN);
                if (!isfinite(s->knots[i][d]) || (d==3 && s->knots[i][d]<=0) || (d==0 && i && s->knots[i][d]<=s->knots[i-1][d])) { json_free(r); goto bad; }
            }
            json_free(r);
        }
    }
    s->max_soft_sigma=json_num(json_get(j,"max_soft_sigma"),NAN);
    if (!isfinite(s->max_soft_sigma) || s->max_soft_sigma<=0 || !s->records || !s->index || s->records_bytes%sizeof(sheet_record) || s->index_bytes%sizeof(sheet_index)) goto bad;
    s->nr=s->records_bytes/sizeof(sheet_record); s->ni=s->index_bytes/sizeof(sheet_index);
    if (s->nr!=json_num(json_get(j,"records"),-1) || s->ni!=json_num(json_get(j,"cells"),-1)) goto bad;
    uint64_t end=0;
    for (size_t i=0;i<s->ni;i++) {
        const sheet_index *ix=&s->index[i]; if (ix->offset!=end || ix->count>s->nr-end) goto bad;
        if (i) { int cmp=0; for (int d=0;d<3 && !cmp;d++) cmp=(ix->cell[d]>s->index[i-1].cell[d])-(ix->cell[d]<s->index[i-1].cell[d]); if (cmp<=0) goto bad; }
        end+=ix->count;
    }
    if (end!=s->nr) goto bad;
    for (size_t i=0;i<s->nr;i++) {
        const sheet_record *r=&s->records[i];
        if (r->kind>4 || r->count<1 || r->count>SHEET_PATH || !isfinite(r->weight) || r->weight<=0 || !isfinite(r->target) ||
            ((r->kind==1 || r->kind==2) && r->count!=2) || (r->kind==0 && r->count!=1)) goto bad;
        for (unsigned p=0;p<r->count;p++) for (int d=0;d<4;d++) if (!isfinite(r->points[p][d])) goto bad;
    }
    json_free(j); return s;
bad:
    fprintf(stderr,"invalid or changed surface_winding geometry: %s\n",manifest);
    json_free(j); sheet_free(s); return nullptr;
}
void sheet_free(sheet_dataset *s) {
    if (!s) return;
    if (s->records) munmap((void *)s->records,s->records_bytes);
    if (s->index) munmap((void *)s->index,s->index_bytes);
    free(s->contacts); free(s->knots); free(s);
}
void sheet_parameters(const sheet_dataset *s,double z,double a[4]) {
    int hi=1; while (hi<s->nk-1 && s->knots[hi][0]<z) hi++;
    double t=(z-s->knots[hi-1][0])/(s->knots[hi][0]-s->knots[hi-1][0]); t=fmax(0,fmin(1,t));
    for (int d=0;d<4;d++) a[d]=(1-t)*s->knots[hi-1][d+1]+t*s->knots[hi][d+1];
}
double sheet_reference(const sheet_dataset *s,const double xyz[3]) {
    double a[4]; sheet_parameters(s,xyz[0],a);
    return hypot(xyz[1]-a[0],xyz[2]-a[1])/a[2]+a[3];
}
void sheet_mask_contacts(const sheet_dataset *s,const int64_t origin[3],int P,uint8_t *ignore) {
    size_t lo=0,hi=s->nc;
    while (lo<hi) { size_t m=lo+(hi-lo)/2; if (s->contacts[m][0]<origin[0]-32) lo=m+1; else hi=m; }
    for (size_t i=lo;i<s->nc && s->contacts[i][0]<origin[0]+P+32;i++) {
        double c[3],r=s->contacts[i][3]; int low[3],high[3],ok=1;
        for (int d=0;d<3;d++) { c[d]=s->contacts[i][d]-origin[d]; low[d]=fmax(0,ceil(c[d]-r)); high[d]=fmin(P-1,floor(c[d]+r)); if (low[d]>high[d]) ok=0; }
        if (!ok) continue;
        for (int z=low[0];z<=high[0];z++) for (int y=low[1];y<=high[1];y++) for (int x=low[2];x<=high[2];x++)
            if ((z-c[0])*(z-c[0])+(y-c[1])*(y-c[1])+(x-c[2])*(x-c[2])<=r*r) ignore[((size_t)z*P+y)*P+x]=1;
    }
}
static uint64_t random64(uint64_t *x) { *x+=0x9e3779b97f4a7c15ull; uint64_t z=*x; z=(z^(z>>30))*0xbf58476d1ce4e5b9ull; z=(z^(z>>27))*0x94d049bb133111ebull; return z^(z>>31); }
sheet_batch *sheet_sample(const sheet_dataset *s,const int64_t origin[3],int P,const int perm[3],const int flip[3],const uint8_t *ct,const uint8_t *surface_target,uint64_t seed) {
    static const unsigned caps[5]={4096,1024,1024,128,1024};
    const sheet_record **chosen[5]; unsigned count[5]={0}; uint64_t seen[5]={0};
    for (int k=0;k<5;k++) chosen[k]=malloc(caps[k]*sizeof *chosen[k]);
    size_t begin=0,end=s->ni;
    const int64_t first_z=origin[0]/s->cell_size,last_z=(origin[0]+P-1)/s->cell_size;
    while (begin<end) { size_t mid=begin+(end-begin)/2; if (s->index[mid].cell[0]<first_z) begin=mid+1; else end=mid; }
    for (size_t c=begin;c<s->ni && s->index[c].cell[0]<=last_z;c++) {
        const sheet_index *ix=&s->index[c]; int match=1;
        for (int d=0;d<3;d++) if ((int64_t)(ix->cell[d]+1)*s->cell_size<=origin[d] || (int64_t)ix->cell[d]*s->cell_size>=origin[d]+P) match=0;
        if (!match) continue;
        for (uint64_t i=ix->offset;i<ix->offset+ix->count;i++) {
            const sheet_record *r=&s->records[i]; int ok=1;
            for (unsigned p=0;p<r->count && ok;p++) {
                int v[3]; for (int d=0;d<3;d++) { double x=r->points[p][d]-origin[d]; if (x<0 || x>=P-1) ok=0; v[d]=(int)lround(x); }
                if (ok && ct && !ct[((size_t)v[0]*P+v[1])*P+v[2]]) ok=0;
                if (ok && r->kind==4 && surface_target && surface_target[((size_t)v[0]*P+v[1])*P+v[2]]>25) ok=0;
            }
            if (!ok) continue;
            unsigned k=r->kind; uint64_t ixr=random64(&seed)%++seen[k];
            if (count[k]<caps[k]) chosen[k][count[k]++]=r;
            else if (ixr<caps[k]) chosen[k][ixr]=r;
        }
    }
    sheet_batch *b=calloc(1,sizeof *b);
    for (int k=0;k<5;k++) for (unsigned j=0;j<count[k];j++) { b->nt++; b->np+=chosen[k][j]->count; }
    b->points=calloc(b->np?b->np:1,sizeof *b->points); b->terms=calloc(b->nt?b->nt:1,sizeof *b->terms);
    size_t nt=0,np=0;
    for (int k=0;k<5;k++) {
        for (unsigned j=0;j<count[k];j++) {
            const sheet_record *r=chosen[k][j]; b->terms[nt++]=(sheet_term){r->kind,r->count,(uint32_t)np,r->weight,r->target};
            for (unsigned p=0;p<r->count;p++) {
                sheet_point *point=&b->points[np++]; double world[3];
                for (int d=0;d<3;d++) { world[d]=r->points[p][d]; double local=r->points[p][perm[d]]-origin[perm[d]]; point->xyz[d]=(float)(flip[d]?P-1-local:local); }
                point->q=r->points[p][3]; point->q0=(float)sheet_reference(s,world);
            }
        }
        free(chosen[k]);
    }
    return b;
}
sheet_batch *sheet_clone(const sheet_batch *b) {
    if (!b) return nullptr;
    sheet_batch *a=malloc(sizeof *a); *a=*b;
    a->points=malloc((b->np?b->np:1)*sizeof *a->points); a->terms=malloc((b->nt?b->nt:1)*sizeof *a->terms);
    memcpy(a->points,b->points,b->np*sizeof *a->points); memcpy(a->terms,b->terms,b->nt*sizeof *a->terms); return a;
}
void sheet_batch_free(sheet_batch *b) { if (b) { free(b->points); free(b->terms); free(b); } }
static double softplus(double x) { return fmax(x,0)+log1p(exp(-fabs(x))); }
double sheet_loss(const sheet_batch *b,const double *v,double *g,double parts[5],float ramp,int variant) {
    const double weights[5]={.25,.25,.25,.1,.1}; double count[5]={0}; memset(parts,0,5*sizeof *parts); memset(g,0,2*b->np*sizeof *g);
    for (size_t i=0;i<b->nt;i++) count[b->terms[i].kind]+=b->terms[i].weight;
    for (size_t i=0;i<b->nt;i++) {
        const sheet_term *t=&b->terms[i]; int k=t->kind; size_t a=t->first,bb=a+1;
        if (variant==0 || (variant==1 && k>=3)) continue;
        double f=t->weight/fmax(count[k],1e-12),wf=f*weights[k]*ramp;
        if (k<=2) {
            double e=k==0?v[2*a+1]-b->points[a].q:v[2*bb+1]-v[2*a+1]-t->target;
            parts[k]+=f*(fabs(e)<=.1?5*e*e:fabs(e)-.05); double dg=fmax(-1,fmin(1,e/.1))*wf;
            g[2*a+1]+=k==0?dg:-dg; if (k) g[2*bb+1]+=dg;
        } else if (k==3) {
            double lo=INFINITY,sum=0; for (unsigned p=0;p<t->count;p++) lo=fmin(lo,v[2*(a+p)]);
            for (unsigned p=0;p<t->count;p++) sum+=exp(-4*(v[2*(a+p)]-lo));
            double score=lo-log(sum/t->count)/4; parts[k]+=f*softplus(-score);
            double dg=-1/(1+exp(fmax(-700,fmin(700,score))))*wf;
            for (unsigned p=0;p<t->count;p++) g[2*(a+p)]+=dg*exp(-4*(v[2*(a+p)]-lo))/sum;
        } else {
            for (unsigned p=0;p<t->count;p++) { double x=v[2*(a+p)]; parts[k]+=f*softplus(x)/t->count; g[2*(a+p)]+=wf/(t->count*(1+exp(fmax(-700,fmin(700,-x))))); }
        }
    }
    double loss=0; for (int k=0;k<5;k++) loss+=parts[k]*weights[k]*ramp; return loss;
}
int sheet_checkpoint(const char *path,char manifest_sha[65],char reference_sha[65]) {
    FILE *f=fopen(path,"rb"); if (!f) return -1; char text[UFSM_CHECKPOINT_HEADER];
    int got=fgets(text,sizeof text,f)!=nullptr; fclose(f);
    if (!got || strncmp(text,"UFSM",4) || !strchr(text,'\n')) return -1;
    json *j=json_parse(text+4,strlen(text+4)); if (!j) return -1;
    const json *s=json_path(j,"extra.sheet"),*task=json_path(j,"extra.task"); int rc=0;
    if (task || s) {
        const char *m=json_str(json_get(s,"geometry_sha256"),""),*r=json_str(json_get(s,"reference_sha256"),"");
        if (strcmp(json_str(task,""),"surface_winding") || json_num(json_get(s,"version"),0)!=1 || strlen(m)!=64 || strlen(r)!=64) rc=-1;
        else { strcpy(manifest_sha,m); strcpy(reference_sha,r); rc=1; }
    }
    json_free(j); return rc;
}
int sheet_checkpoint_options(const char *path,int *variant,int *schedule_start) {
    FILE *f=fopen(path,"rb"); if (!f) return -1;
    char text[UFSM_CHECKPOINT_HEADER]; int got=fgets(text,sizeof text,f)!=nullptr; fclose(f);
    if (!got || strncmp(text,"UFSM",4) || !strchr(text,'\n')) return -1;
    json *j=json_parse(text+4,strlen(text+4)); if (!j) return -1;
    double v=json_num(json_path(j,"extra.sheet.variant"),NAN),s=json_num(json_path(j,"extra.sheet.schedule_start"),NAN);
    int bad=!isfinite(v) || !isfinite(s) || v<0 || v>2 || floor(v)!=v || s<0 || s>2147483647 || floor(s)!=s;
    if (!bad) { *variant=(int)v; *schedule_start=(int)s; }
    json_free(j); return bad?-1:0;
}
