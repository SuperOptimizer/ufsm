/* Whole-network gradients and buffer rebuilding for larger final-decoder up-gradient chunks.
   Compare round-to-nearest exactly and stochastic differences against another baseline seed. */
#include "../src/unet.c"
static int failures;
typedef struct {char name[64];size_t off,len;} segment;
static segment segments[128];static int nsegment;
static void add_segment(const char *name,size_t off,size_t len){segment *s=&segments[nsegment++];snprintf(s->name,sizeof s->name,"%s",name);s->off=off;s->len=len;}
static void probe_add_block(const char *name,const block *b){char label[64];
 snprintf(label,sizeof label,"%s.c1",name);add_segment(label,b->c1.w,(size_t)b->c1.cin*b->c1.cout*27);
 snprintf(label,sizeof label,"%s.c2",name);add_segment(label,b->c2.w,(size_t)b->c2.cin*b->c2.cout*27);
 snprintf(label,sizeof label,"%s.gn1",name);add_segment(label,b->n1.gamma,2*(size_t)b->c1.cout);
 snprintf(label,sizeof label,"%s.gn2",name);add_segment(label,b->n2.gamma,2*(size_t)b->c2.cout);
}
static uint64_t rng=123;
static float random_value(void) {rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;return (float)((rng>>11)*0x1.0p-53*2-1);}
static void run(const unet_cfg *cfg,shape5 xs,const float *x,const float *gy,int wide,unsigned seed,float *logits,float *grad) {
 unet_set_wide_up_grad(wide);
 nn_set_sr_step(seed);
 unet *u=unet_create(cfg);unet_init(u,7);
 size_t planned=unet_train_bytes(u,xs);
 const float *lg=unet_forward(u,x,xs,1);
 if(u->act_bytes!=planned){fprintf(stderr,"dry build differs from forward buffers\n");failures++;}
 nn_d2h(logits,lg,shape_numel(unet_out_shape(u,xs))*4);
 unet_zero_grad(u);unet_backward(u,gy);unet_grad_d2h(u,grad);
 shape5 up=xs;up.c=cfg->widths[1];
 printf("wide=%d seed=%u cap=%zu required_up=%zu activation_bytes=%zu\n",wide,seed,u->gB_cap[0],nn_mx8_bytes(up),unet_activation_bytes(u));
 if(wide && u->gB_cap[0]<nn_mx8_bytes(up)) failures++;

 nn_sync();const char *e=nn_check();if(e){fprintf(stderr,"%s\n",e);exit(2);}
 unet_free(u);
}
static double rel(const float*a,const float*b,size_t n) {double d2=0,r2=0;for(size_t i=0;i<n;i++){double d=(double)a[i]-b[i];d2+=d*d;r2+=(double)b[i]*b[i];if(!isfinite(a[i])||!isfinite(b[i]))failures++;}return sqrt(d2/fmax(r2,1e-300));}
int main(void) {
 if(nn_init(0))return 2;
 setenv("UFSM_F4_WGRAD","1",1);nn_set_f16(1);nn_set_grad_scale(1024);nn_set_gn_stored(1);int sr=getenv("UFSM_TEST_SR")?atoi(getenv("UFSM_TEST_SR")):1;nn_set_sr(sr);
 if(nn_set_prec_policy("all=fp4:fp4:fp4,enc0.c1=fp16"))return 2;
 unet_set_act_mx4(1);unet_set_input_prec(8);unet_set_input_mx(1);unet_set_grad_mx8(1);unet_set_recompute(1);unet_set_chunk_up(2);unet_set_lean(2);
 unet_cfg cfg={4,{16,32,64,80},4,2,8,1};int P=getenv("UFSM_TEST_P")?atoi(getenv("UFSM_TEST_P")):32;shape5 xs={P>32?1:2,4,P,P+8,P+16};
 unet *probe=unet_create(&cfg);size_t np=unet_nparams(probe),nl=shape_numel(unet_out_shape(probe,xs));
 for(int l=0;l<cfg.nlev;l++){char label[64];snprintf(label,sizeof label,"enc%d",l);probe_add_block(label,&probe->enc[l]);if(l+1<cfg.nlev){snprintf(label,sizeof label,"dec%d",l);probe_add_block(label,&probe->dec[l]);}}
 unet_free(probe);
 size_t nx=shape_numel(xs);float *hx=malloc(nx*4),*hgy=malloc(nl*4),*dx=nn_malloc(nx*4),*dg=nn_malloc(nl*4);
 for(size_t i=0;i<nx;i++)hx[i]=random_value();
 for(size_t i=0;i<nl;i++)hgy[i]=random_value()*0.01f;
 nn_h2d(dx,hx,nx*4);nn_h2d(dg,hgy,nl*4);
 float *logs[5],*grads[5];int modes[]={0,0,1,1,0};unsigned seeds[]={1,2,1,2,1};
 for(int j=0;j<5;j++){logs[j]=malloc(nl*4);grads[j]=malloc(np*4);run(&cfg,xs,dx,dg,modes[j],seeds[j],logs[j],grads[j]);}
 for(int j=2;j<4;j++) {
  int ref=j-2;double f=rel(logs[j],logs[ref],nl),g=rel(grads[j],grads[ref],np),noise=rel(grads[1],grads[0],np);
  for(int t=0;t<nsegment;t++){segment sg=segments[t];double d=rel(grads[j]+sg.off,grads[ref]+sg.off,sg.len),n=rel(grads[1]+sg.off,grads[0]+sg.off,sg.len);int pass=sr?d<1.5*n+1e-3:d<1e-5;printf("  %s grad_rel=%.6g seed_rel=%.6g %s\n",sg.name,d,n,pass?"ok":"FAIL");failures+=!pass;}
  int ok=f<1e-5 && (sr?g<1.5*noise+1e-3:g<1e-5);printf("seed=%u logits_rel=%.6g grad_rel=%.6g baseline_seed_rel=%.6g %s\n",seeds[j],f,g,noise,ok?"ok":"FAIL");failures+=!ok;
 }
 double repeat=rel(grads[4],grads[0],np);printf("repeat narrow grad_rel=%.6g\n",repeat);failures+=repeat>=1e-5;
 nn_set_sr(0);unet_set_wide_up_grad(0);unet *reused=unet_create(&cfg);unet_init(reused,7);
 const int transitions[]={0,1,0,1};
 for(int k=0;k<4;k++){
  unet_set_wide_up_grad(transitions[k]);nn_set_sr_step(1);
  const float *lg=unet_forward(reused,dx,xs,1);nn_d2h(logs[4],lg,nl*4);
  unet_zero_grad(reused);unet_backward(reused,dg);unet_grad_d2h(reused,grads[4]);
  float *want_l=malloc(nl*4),*want_g=malloc(np*4);run(&cfg,xs,dx,dg,transitions[k],1,want_l,want_g);
  double le=rel(logs[4],want_l,nl),ge=rel(grads[4],want_g,np);int pass=le<1e-5 && ge<1e-5;
  printf("reused transition wide=%d logits_rel=%.6g grad_rel=%.6g %s\n",transitions[k],le,ge,pass?"ok":"FAIL");failures+=!pass;free(want_l);free(want_g);
 }
 unet_free(reused);unet_set_wide_up_grad(0);
 for(int j=0;j<5;j++){free(logs[j]);free(grads[j]);}free(hx);free(hgy);nn_free(dx);nn_free(dg);
 printf("wide-up gradients and buffer transitions: %s\n",failures?"FAIL":"ok");return failures!=0;
}
