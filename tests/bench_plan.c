/* bench_plan: the training network's planned device bytes (unet_train_bytes, a dry build: no device allocation beyond the context) for
   any architecture and memory mode, the way train's planner builds them (--fp4 2: MX-fp4 activations and gradients, the fp16 stem,
   MX-fp4 input), without a sampler or sources. Env:
     BM_WIDTHS "16,64,96,128,160,192"   encoder widths        BM_DECW "0,16"   decoder widths (0 = the level's)
     BM_COUT 1                          head outputs          BM_P "512,640,768"  windows (cubes)
     BM_MODE "4,5"                      planner modes: 4 = MX gradients, chunked, lean 2, shared encoder a1 (host offloads on);
                                        5 = the same with the three level-0 buffers; 3 = mode 4 without the shared encoder a1;
                                        14 = mode 4 without host offloads (--mem fast's pass 0)
     BM_HEADSLAB 0 (1: train's head slabs, no whole logits)
     BM_DOWNNORM 1, BM_SCALECOND 1      as wind96             BM_TAIL bytes per voxel of the batch tail (targets + mask, default 2)
   UFSM_MEM_REPORT=1 / 2 add the dry build's breakdown. Prints one line per (P, mode): bytes, B per voxel, the offload bytes, l3b used. */
#include "nn.h"
#include "unet.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int ints(const char *s, int *v, int n) { int k = 0; char *t = strdup(s); for (char *q = strtok(t, ","); q && k < n; q = strtok(nullptr, ",")) v[k++] = atoi(q); free(t); return k; }

int main(void) {
    const char *we = getenv("BM_WIDTHS") ? getenv("BM_WIDTHS") : "16,64,96,128,160,192";
    unet_cfg cfg = {0};
    cfg.nlev = ints(we, cfg.widths, UNET_MAXLEV);
    if (getenv("BM_DECW")) ints(getenv("BM_DECW"), cfg.dec_widths, UNET_MAXLEV);
    cfg.cin = 4; cfg.cout = getenv("BM_COUT") ? atoi(getenv("BM_COUT")) : 1; cfg.G = 8;
    cfg.down_norm = getenv("BM_DOWNNORM") ? atoi(getenv("BM_DOWNNORM")) : 1; cfg.scale_cond = getenv("BM_SCALECOND") ? atoi(getenv("BM_SCALECOND")) : 1;
    int Ps[16], modes[16];
    const int nP = ints(getenv("BM_P") ? getenv("BM_P") : "512,640,768", Ps, 16), nm = ints(getenv("BM_MODE") ? getenv("BM_MODE") : "4,5", modes, 16);
    const double tail = getenv("BM_TAIL") ? atof(getenv("BM_TAIL")) : 2;
    nn_init(0);
    nn_set_tf32(1); nn_set_act_bf16(1); nn_set_grad_bf16(1); nn_set_f16(1); nn_set_grad_scale(1024); nn_set_sr(1); nn_set_gn_stored(1);
    nn_set_nlev(cfg.nlev);
    if (nn_set_prec_policy("all=fp4:fp4:fp4,enc0.c1=fp16")) return 2;
    setenv("UFSM_F4_WGRAD", "1", 1);
    unet_set_act_mx4(1); unet_set_grad_mx8(1); unet_set_grad_mx4(1); unet_set_input_mx(1); unet_set_input_prec(4);
    unet *u = unet_create(&cfg);
    unet_set_fp16_logits(u, 1);
    if (getenv("BM_HEADSLAB") && atoi(getenv("BM_HEADSLAB"))) unet_set_head_ext(u, 1);   /* train's head slabs: no whole-window logits */
    printf("widths %s dec %s cout %d: %.2f M parameters\n", we, getenv("BM_DECW") ? getenv("BM_DECW") : "-", cfg.cout, unet_nparams(u) / 1e6);
    for (int a = 0; a < nP; a++) for (int b = 0; b < nm; b++) {
        const int P = Ps[a], m = modes[b];
        const shape5 xs = {1, cfg.cin, P, P, P};
        const double nv = (double)P * P * P;
        unet_set_host_offloads(m != 14);
        unet_set_chunk_up(2); unet_set_recompute(1); unet_set_lean(2); unet_set_share_enc_a1(m != 3); unet_set_l0_3buf(m == 5);
        unet_set_batch_bytes(u, 0, (size_t)(tail * nv) + 512);
        const size_t tb = unet_train_bytes(u, xs);
        const size_t off = unet_last_offload_bytes(); const int l3 = unet_last_dry_l3b();
        const size_t ws = unet_ws_plan(u, xs, 0, 0, nullptr);   /* (the kernels' workspaces this mode plans: train's planner adds them) */
        printf("P %4d mode %2d: %6.2f GB  %5.1f B/voxel  + 1/14 margin + kernel workspaces %.2f GB = %.2f GB  host offloads %5.2f GB each way  l3b %d\n",
               P, m, tb / 1e9, tb / nv, ws / 1e9, (tb + tb / 14 + ws) / 1e9, off / 1e9, l3);
        fflush(stdout);
    }
    unet_free(u);
    return 0;
}
