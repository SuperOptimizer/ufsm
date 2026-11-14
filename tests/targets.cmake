# The test, bench and tool binaries (CMakeLists.txt: ufsm_exe). One call per binary:
#   ufsm_exe(NAME [SUITE] SOURCES... [OBJS obj... | NN | LP | CORE] [LIBS LDLIBS | CUDA | lib...] [DEFS d...] [INC CUDA | BUILD] [DEPS t...])
# SUITE puts it into suite-bins (and needs a line in tests/suite.txt: cmake --build build-verify --target suite-check).
# OBJS names src objects (fit = src/fit.c), NN / LP = every nn_*.cu / lp_*.cu object, CORE = every src/*.c object but main.
# LIBS: LDLIBS = libcurl libzstd blosc zlib libcrypto libjpeg libm pthread, CUDA = cudart + cublasLt, else -l<name>.
# DEPS: binaries the test runs (built first).

ufsm_exe(aug_preview tools/aug_preview.c OBJS scannorm edt band sheet ct_augment scan_augment spatial_augment crush scrollaug cover json
         store zarr3 sources band_onfly fuse fiberraster afv regmap srpair zarr2 tiff z3w hf zipr view reg sample_emb embz LIBS LDLIBS)
ufsm_exe(aug_render tests/aug_render.c OBJS scannorm edt band sheet ct_augment scan_augment spatial_augment crush scrollaug cover json store
         zarr3 sources band_onfly fuse fiberraster afv regmap srpair zarr2 tiff z3w hf zipr view reg sample_emb embz imgops jpegpar tiffw
         tfx surfcomp LIBS LDLIBS)
ufsm_exe(bench_attn tests/bench_attn.c OBJS NN LP LIBS LDLIBS CUDA)
ufsm_exe(bench_conv tests/bench_conv.c OBJS NN LP LIBS m CUDA)
ufsm_exe(bench_f4conv tests/bench_f4conv.c OBJS NN LP LIBS m CUDA)
ufsm_exe(bench_gn tests/bench_gn.c OBJS NN LP LIBS m CUDA)
ufsm_exe(bench_head tests/bench_head.c OBJS NN LP LIBS m CUDA)
ufsm_exe(bench_head2 tests/bench_head2.c OBJS NN LP LIBS m CUDA)
ufsm_exe(bench_l0_ring tests/bench_l0_ring.c OBJS unet NN LP LIBS m CUDA)
ufsm_exe(bench_l0misc tests/bench_l0misc.c OBJS NN LP LIBS m CUDA)
ufsm_exe(bench_lp tests/bench_lp.c OBJS NN LP LIBS m CUDA)
ufsm_exe(bench_mem tests/bench_mem.c OBJS unet NN LP LIBS m CUDA)
ufsm_exe(bench_s2 tests/bench_s2.c OBJS unet NN LP LIBS m CUDA)
ufsm_exe(bench_muonc tests/bench_muonc.c OBJS unet NN LP LIBS m CUDA)
ufsm_exe(bench_muonx tests/bench_muonx.c OBJS unet NN LP LIBS m CUDA)
ufsm_exe(bench_read tests/bench_read.c OBJS sample scannorm sample_emb embz edt band sheet ct_augment scan_augment spatial_augment crush
         scrollaug cover sources band_onfly fuse fiberraster afv reg regmap zarr2 srpair view zarr3 store json z3w tiff zipr hf NN LP LIBS m
         curl zstd blosc z crypto CUDA)
ufsm_exe(bench_sampler tests/bench_sampler.c OBJS sample scannorm sample_emb embz edt band sheet ct_augment scan_augment spatial_augment
         crush scrollaug cover sources band_onfly fuse fiberraster afv reg regmap zarr2 srpair view zarr3 store json z3w tiff zipr hf NN LP
         LIBS m curl zstd blosc z crypto CUDA)
ufsm_exe(bench_votes tests/bench_votes.c OBJS zarr3 store json LIBS LDLIBS)
ufsm_exe(bench_warp tests/bench_warp.c OBJS scannorm edt band sheet ct_augment scan_augment spatial_augment crush scrollaug cover json store
         zarr3 sources band_onfly fuse fiberraster afv regmap srpair zarr2 tiff z3w hf zipr view reg sample_emb embz LIBS LDLIBS)
ufsm_exe(bench_wgrad4 tests/bench_wgrad4.c OBJS NN LP LIBS m CUDA)
ufsm_exe(crush_anim tests/crush_anim.c OBJS scannorm edt band sheet ct_augment scan_augment spatial_augment crush scrollaug cover json store
         zarr3 sources band_onfly fuse fiberraster afv regmap srpair zarr2 tiff z3w hf zipr view reg sample_emb embz imgops jpegpar tiffw
         tfx surfcomp LIBS LDLIBS)
ufsm_exe(crush_render tests/crush_render.c OBJS scannorm edt band sheet ct_augment scan_augment spatial_augment crush scrollaug cover json
         store zarr3 sources band_onfly fuse fiberraster afv regmap srpair zarr2 tiff z3w hf zipr view reg sample_emb embz imgops jpegpar
         tiffw tfx surfcomp LIBS LDLIBS)
ufsm_exe(fwd_nan tests/fwd_nan.c OBJS unet NN LP LIBS m CUDA)
ufsm_exe(make_pipeline_fixture SUITE tests/make_pipeline_fixture.c OBJS z3w LIBS LDLIBS)
ufsm_exe(make_wind_fixture SUITE tests/make_wind_fixture.c OBJS z3w LIBS LDLIBS)
ufsm_exe(phase_pack src/phasepack.c OBJS store zarr3 json z3w LIBS LDLIBS DEFS PHASE_PACK_MAIN)
ufsm_exe(prec_sweep tests/prec_sweep.c OBJS unet NN LP LIBS m CUDA)
ufsm_exe(prof_infer tests/prof_infer.c OBJS unet NN LP LIBS m CUDA)
ufsm_exe(scrollaug_anim tests/scrollaug_anim.c OBJS scannorm edt band sheet ct_augment scan_augment spatial_augment crush scrollaug cover
         json store zarr3 sources band_onfly fuse fiberraster afv regmap srpair zarr2 tiff z3w hf zipr view reg sample_emb embz imgops
         jpegpar tiffw tfx surfcomp LIBS LDLIBS)
ufsm_exe(store_bench src/storebench.c OBJS store zarr3 json LIBS LDLIBS DEFS STORE_BENCH_MAIN)
ufsm_exe(surface_dedup tools/surface_dedup.c OBJS tiff LIBS m z)
ufsm_exe(test_afv SUITE tests/test_afv.c OBJS afv json LIBS LDLIBS)
ufsm_exe(test_annot SUITE tests/test_annot.c OBJS annot z3w zarr3 store json tfx imgops jpegpar tiffw tiff surfcomp LIBS LDLIBS)
ufsm_exe(test_attn SUITE tests/test_attn.c OBJS NN LP LIBS LDLIBS CUDA)
ufsm_exe(test_attn_dfm SUITE tests/test_attn_dfm.c OBJS NN LP LIBS LDLIBS CUDA)
ufsm_exe(test_attn_h16 SUITE tests/test_attn_h16.c OBJS NN LP LIBS LDLIBS CUDA)
ufsm_exe(test_attn_ray SUITE tests/test_attn_ray.c OBJS NN LP LIBS LDLIBS CUDA)
ufsm_exe(test_aug_preset SUITE tests/test_aug_preset.c OBJS scannorm edt band sheet ct_augment scan_augment spatial_augment crush scrollaug
         cover json store zarr3 sources band_onfly fuse fiberraster afv regmap srpair zarr2 tiff z3w hf zipr view reg sample_emb embz LIBS
         LDLIBS)
ufsm_exe(test_aug_z SUITE tests/test_aug_z.c OBJS scannorm edt band sheet ct_augment scan_augment spatial_augment crush scrollaug cover json
         store zarr3 sources band_onfly fuse fiberraster afv regmap srpair zarr2 tiff z3w hf zipr view reg sample_emb embz NN LP LIBS LDLIBS
         CUDA)
ufsm_exe(test_band_pair SUITE tests/test_band_pair.c OBJS band nn_band LIBS LDLIBS CUDA INC CUDA)
ufsm_exe(test_checkpoint SUITE tests/test_checkpoint.c OBJS unet NN LP LIBS m CUDA)
ufsm_exe(test_checkpoint_runtime SUITE tests/test_checkpoint_runtime.c OBJS checkpoint json LIBS m)
ufsm_exe(test_cond SUITE tests/test_cond.c OBJS scannorm edt band sheet ct_augment scan_augment spatial_augment crush scrollaug cover json
         store zarr3 sources band_onfly fuse fiberraster afv regmap srpair zarr2 tiff z3w hf zipr view reg sample_emb embz LIBS LDLIBS)
ufsm_exe(test_coreline SUITE tests/test_coreline.c OBJS coreline z3w fit fit_emb embz fit_ct l1pot json zarr3 store NN LP LIBS LDLIBS CUDA)
ufsm_exe(test_countfield SUITE tests/test_countfield.c OBJS countfield ctcal fit fit_emb embz fit_ct l1pot coreline json zarr3 store z3w NN
         LP LIBS LDLIBS CUDA)
ufsm_exe(test_cover SUITE tests/test_cover.c OBJS cover json LIBS crypto m)
ufsm_exe(test_crush SUITE tests/test_crush.c OBJS scannorm edt band sheet ct_augment scan_augment spatial_augment crush scrollaug cover json
         store zarr3 sources band_onfly fuse fiberraster afv regmap srpair zarr2 tiff z3w hf zipr view reg sample_emb embz LIBS LDLIBS)
ufsm_exe(test_ct_augment SUITE tests/test_ct_augment.c OBJS ct_augment LIBS m)
ufsm_exe(test_density_lambda SUITE tests/test_density_lambda.c OBJS unet json zarr3 store z3w NN LP LIBS LDLIBS CUDA DEPS ufsm
         make_wind_fixture)
ufsm_exe(test_dino_loss SUITE tests/test_dino_loss.c OBJS NN LP LIBS LDLIBS CUDA)
ufsm_exe(test_dino_pts SUITE tests/test_dino_pts.c LIBS LDLIBS)
ufsm_exe(test_dinor SUITE tests/test_dinor.c OBJS NN LP LIBS LDLIBS CUDA)
ufsm_exe(test_distill SUITE tests/test_distill.c OBJS sample_emb embz z3w spatial_augment json NN LP LIBS LDLIBS CUDA)
ufsm_exe(test_edt SUITE tests/test_edt.c OBJS edt LIBS pthread m)
ufsm_exe(test_eval SUITE tests/test_eval.c OBJS scannorm edt band sheet ct_augment scan_augment spatial_augment crush scrollaug cover json
         store zarr3 sources band_onfly fuse fiberraster afv regmap srpair zarr2 tiff z3w hf zipr view reg sample_emb embz LIBS LDLIBS)
ufsm_exe(test_faces SUITE tests/test_faces.c OBJS faces LIBS LDLIBS)
ufsm_exe(test_faces_gpu SUITE tests/test_faces.c OBJS faces nn_faces LIBS LDLIBS CUDA DEFS FACES_GPU)
ufsm_exe(test_fblend SUITE tests/test_fblend.c OBJS fblend fit fit_emb embz fit_ct l1pot coreline json zarr3 store z3w NN LP LIBS LDLIBS
         CUDA)
ufsm_exe(test_fib_dir SUITE tests/test_fib_dir.c OBJS NN LP LIBS m CUDA)
ufsm_exe(test_fiber_link SUITE tests/test_fiber_link.c OBJS fiberlink afv json LIBS LDLIBS)
ufsm_exe(test_fiberjudge SUITE tests/test_fiberjudge.c OBJS fiberjudge wrapdump fitphase afv fitjoin phasegate fit fit_emb fit_ct l1pot
         coreline tfx surfcomp tiffw scannorm edt band sheet ct_augment scan_augment spatial_augment crush scrollaug cover json store zarr3
         sources band_onfly fuse fiberraster regmap srpair zarr2 tiff z3w hf zipr view reg sample_emb embz NN LP LIBS LDLIBS CUDA)
ufsm_exe(test_fiberraster SUITE tests/test_fiberraster.c OBJS fiberraster nn_fibre afv reg regmap z3w zarr3 zarr2 store json LIBS LDLIBS
         CUDA)
ufsm_exe(test_fibertrace SUITE tests/test_fibertrace.c OBJS fibertrace ftaxis nn_ftrace afv zarr2 zarr3 store json LIBS LDLIBS CUDA)
ufsm_exe(test_fibertrace_dyad SUITE tests/test_fibertrace_dyad.c OBJS fibertrace ftaxis nn_ftrace afv zarr2 zarr3 z3w store json LIBS LDLIBS
         CUDA)
ufsm_exe(test_fibhead SUITE tests/test_fibhead.c OBJS NN LP LIBS m CUDA)
ufsm_exe(test_fibn SUITE tests/test_fibn.c OBJS NN LP LIBS m CUDA)
ufsm_exe(test_fields_pack SUITE tests/test_fields_pack.c OBJS fieldspack repack fit fit_emb embz fit_ct l1pot coreline json zarr3 store z3w
         NN LP LIBS LDLIBS CUDA)
ufsm_exe(test_fields_phase SUITE tests/test_fields_phase.c OBJS fieldsphase fit fit_emb embz fit_ct l1pot coreline json zarr3 store z3w NN
         LP LIBS LDLIBS CUDA)
ufsm_exe(test_fit SUITE tests/test_fit.c OBJS fit fit_emb embz z3w fit_ct l1pot coreline json zarr3 store NN LP LIBS LDLIBS CUDA)
ufsm_exe(test_fit_emb SUITE tests/test_fit_emb.c OBJS fit_emb embz z3w fit fit_ct l1pot coreline json zarr3 store NN LP LIBS LDLIBS CUDA)
ufsm_exe(test_fit_join SUITE tests/test_fit_join.c OBJS fitjoin afv phasegate fit fit_emb embz z3w fit_ct l1pot coreline json zarr3 store NN
         LP LIBS LDLIBS CUDA)
ufsm_exe(test_fit_lat SUITE tests/test_fit_lat.c OBJS NN LP LIBS m CUDA)
ufsm_exe(test_fit_w4 SUITE tests/test_fit_w4.c OBJS fit fit_emb embz z3w fit_ct l1pot coreline json zarr3 store NN LP LIBS LDLIBS CUDA)
ufsm_exe(test_flatten SUITE tests/test_flatten.c OBJS flatten imgops jpegpar tiffw tiff LIBS LDLIBS)
ufsm_exe(test_flatten_gpu SUITE tests/test_flatten.c OBJS flatten imgops jpegpar tiffw tiff nn_flat LIBS LDLIBS CUDA DEFS FLAT_GPU)
ufsm_exe(test_formats SUITE tests/test_formats.c OBJS CORE NN LP LIBS LDLIBS CUDA)
ufsm_exe(test_fstem SUITE tests/test_fstem.c OBJS NN LP LIBS m CUDA)
ufsm_exe(test_slab_tg SUITE tests/test_slab_tg.c OBJS NN LP LIBS m CUDA)
ufsm_exe(test_ftrace_gpu SUITE tests/test_ftrace_gpu.c OBJS ftaxis nn_ftrace LIBS m CUDA INC CUDA)
ufsm_exe(test_fuse SUITE tests/test_fuse.c src/fuse.c LIBS m DEPS)
ufsm_exe(test_fused SUITE tests/test_fused.c OBJS NN LP LIBS m CUDA)
ufsm_exe(test_geom_check SUITE tests/test_geom_check.c OBJS geomcheck sources band_onfly fuse fiberraster afv reg regmap zarr2 band edt
         srpair json zarr3 store z3w tiffw tiff tfx surfcomp LIBS LDLIBS)
ufsm_exe(test_gfilm SUITE tests/test_gfilm.c OBJS unet NN LP LIBS m CUDA)
ufsm_exe(test_gn_contract SUITE tests/test_gn_contract.c OBJS NN LP LIBS m CUDA)
ufsm_exe(test_gn_ema SUITE tests/test_gn_ema.c OBJS unet NN LP LIBS m CUDA)
ufsm_exe(test_gn_frozen SUITE tests/test_gn_frozen.c OBJS unet NN LP LIBS m CUDA)
ufsm_exe(test_gpu_aug SUITE tests/test_gpu_aug.c OBJS scannorm edt band sheet ct_augment scan_augment spatial_augment crush scrollaug cover
         json store zarr3 sources band_onfly fuse fiberraster afv regmap srpair zarr2 tiff z3w hf zipr view reg sample_emb embz NN LP LIBS
         LDLIBS CUDA)
ufsm_exe(test_gpu_input SUITE tests/test_gpu_input.c OBJS sample scannorm sample_emb embz edt band sheet ct_augment scan_augment
         spatial_augment crush scrollaug cover sources band_onfly fuse fiberraster afv reg regmap zarr2 srpair view zarr3 store json z3w
         tiff zipr hf NN LP LIBS m curl zstd blosc z crypto CUDA)
ufsm_exe(test_gpudec SUITE tests/test_gpudec.c OBJS zarr3 store json z3w nn_gpudec LIBS LDLIBS CUDA INC CUDA)
ufsm_exe(test_grad_mx4 SUITE tests/test_grad_mx4.c OBJS unet NN LP LIBS m CUDA)
ufsm_exe(test_gram SUITE tests/test_gram.c OBJS NN LP LIBS m CUDA)
ufsm_exe(test_grid_sparse SUITE tests/test_grid_sparse.c OBJS fit fit_emb embz z3w fit_ct l1pot coreline json zarr3 store NN LP LIBS LDLIBS
         CUDA)
ufsm_exe(test_gyrec SUITE tests/test_gyrec.c OBJS NN LP LIBS m CUDA)
ufsm_exe(test_half SUITE tests/test_half.c OBJS unet NN LP LIBS m CUDA)
ufsm_exe(test_head32 SUITE tests/test_head32.c OBJS unet NN LP LIBS m CUDA)
ufsm_exe(test_heads SUITE tests/test_heads.c LIBS m DEPS ufsm make_wind_fixture)
ufsm_exe(test_http_reader SUITE tests/test_http_reader.c OBJS store zarr3 json LIBS LDLIBS)
ufsm_exe(test_hybrid_identity SUITE tests/test_hybrid_identity.c OBJS unet NN LP LIBS LDLIBS CUDA)
ufsm_exe(test_imgops SUITE tests/test_imgops.c OBJS imgops jpegpar tiffw tiff LIBS LDLIBS)
ufsm_exe(test_infer_buffers SUITE tests/test_infer_buffers.c OBJS unet NN LP LIBS m CUDA)
ufsm_exe(test_infer_mem SUITE tests/test_infer_mem.c OBJS unet split NN LP LIBS m CUDA)
ufsm_exe(test_init_lr SUITE tests/test_init_lr.c OBJS unet NN LP LIBS LDLIBS CUDA)
ufsm_exe(test_ink_air SUITE tests/test_ink_air.c OBJS CORE NN LP LIBS LDLIBS CUDA)
ufsm_exe(test_ink_gate SUITE tests/test_ink_gate.c OBJS inkgate json zarr3 store z3w LIBS LDLIBS)
ufsm_exe(test_ink_mil SUITE tests/test_ink_mil.c OBJS CORE NN LP LIBS LDLIBS CUDA)
ufsm_exe(test_ink_mil_check SUITE tests/test_ink_mil_check.c OBJS z3w tiffw LIBS LDLIBS DEPS ufsm make_wind_fixture)
ufsm_exe(test_ink_mil_train SUITE tests/test_ink_mil_train.c OBJS CORE NN LP LIBS LDLIBS CUDA DEPS ufsm make_wind_fixture)
ufsm_exe(test_ink_pool SUITE tests/test_ink_pool.c OBJS CORE NN LP LIBS LDLIBS CUDA)
ufsm_exe(test_ink_splat SUITE tests/test_ink_splat.c OBJS CORE NN LP LIBS LDLIBS CUDA)
ufsm_exe(test_jpegpar SUITE tests/test_jpegpar.c OBJS jpegpar imgops tiffw tiff LIBS LDLIBS)
ufsm_exe(test_gns SUITE tests/test_gns.c LIBS m)
ufsm_exe(test_json SUITE tests/test_json.c OBJS json LIBS LDLIBS)
ufsm_exe(test_l0_norm SUITE tests/test_l0_norm.c OBJS unet NN LP LIBS m CUDA)
ufsm_exe(test_l0_slab SUITE tests/test_l0_slab.c OBJS unet NN LP LIBS m CUDA)
ufsm_exe(test_l0_stream SUITE tests/test_l0_stream.c OBJS unet NN LP LIBS m CUDA)
ufsm_exe(test_l3_sched SUITE tests/test_l3_sched.c OBJS unet NN LP LIBS m CUDA)
ufsm_exe(test_lat SUITE tests/test_lat.c OBJS NN LP LIBS LDLIBS CUDA)
ufsm_exe(test_lat_conv0 SUITE tests/test_lat_conv0.c OBJS NN LP LIBS m CUDA)
ufsm_exe(test_lat_fuse SUITE tests/test_lat_fuse.c OBJS NN LP LIBS LDLIBS CUDA)
ufsm_exe(test_lat_head SUITE tests/test_lat_head.c OBJS unet NN LP LIBS m CUDA)
ufsm_exe(test_lathead SUITE tests/test_lathead.c OBJS NN LP LIBS m CUDA)
ufsm_exe(test_latvol SUITE tests/test_latvol.c OBJS latvol nn_latvol LIBS LDLIBS CUDA INC CUDA)
ufsm_exe(test_latvol_nolink SUITE tests/test_latvol_nolink.c OBJS store zarr3 json LIBS LDLIBS)
ufsm_exe(test_lldec SUITE tests/test_lldec.c OBJS band_onfly fuse band edt nn_band z3w zarr3 store json LIBS LDLIBS CUDA)
ufsm_exe(test_mesh_phase SUITE tests/test_mesh_phase.c OBJS CORE NN LP LIBS LDLIBS CUDA DEPS ufsm)
ufsm_exe(test_muon SUITE tests/test_muon.c OBJS NN LP LIBS m CUDA)
ufsm_exe(test_muonc SUITE tests/test_muonc.c OBJS NN LP LIBS m CUDA)
ufsm_exe(test_muonx SUITE tests/test_muonx.c OBJS NN LP LIBS m CUDA)
ufsm_exe(test_admuon SUITE tests/test_admuon.c OBJS NN LP LIBS m CUDA)
ufsm_exe(test_mx SUITE tests/test_mx.c OBJS NN LP LIBS m CUDA)
ufsm_exe(test_mx4 SUITE tests/test_mx4.c OBJS NN LP LIBS m CUDA)
ufsm_exe(test_mx_sw SUITE tests/test_mx_sw.c OBJS NN LP LIBS m CUDA)
ufsm_exe(test_nn SUITE tests/test_nn.c OBJS NN LP LIBS m CUDA)
ufsm_exe(test_optimizer_owners SUITE tests/test_optimizer_owners.c OBJS NN LP LIBS m CUDA)
ufsm_exe(test_pack_tm SUITE tests/test_pack_tm.c OBJS sample scannorm sample_emb embz edt band sheet ct_augment scan_augment spatial_augment
         crush scrollaug cover sources band_onfly fuse fiberraster afv reg regmap zarr2 srpair view zarr3 store json z3w tiff zipr hf NN LP
         LIBS m curl zstd blosc z crypto CUDA)
ufsm_exe(test_pairacc SUITE tests/test_pairacc.c OBJS pairacc LIBS m)
ufsm_exe(test_papyrus SUITE tests/test_papyrus.c LIBS m DEPS)
ufsm_exe(test_patchid SUITE tests/test_patchid.c OBJS patchid pairacc tfx surfcomp tiffw scannorm edt band sheet ct_augment scan_augment
         spatial_augment crush scrollaug cover json store zarr3 sources band_onfly fuse fiberraster afv regmap srpair zarr2 tiff z3w hf zipr
         view reg sample_emb embz NN LP LIBS LDLIBS CUDA)
ufsm_exe(test_patchsolve SUITE tests/test_patchsolve.c OBJS patchsolve tracebox sheettrace imgops jpegpar l1pot tiffw scannorm edt band
         sheet ct_augment scan_augment spatial_augment crush scrollaug cover json store zarr3 sources band_onfly fuse fiberraster afv regmap
         srpair zarr2 tiff z3w hf zipr view reg sample_emb embz NN LP LIBS LDLIBS CUDA)
ufsm_exe(test_pgrad SUITE tests/test_pgrad.c OBJS NN LP LIBS m CUDA)
ufsm_exe(test_phase_check SUITE tests/test_phase_check.c OBJS phase_check LIBS m)
ufsm_exe(test_phase_interp SUITE tests/test_phase_interp.c OBJS sample scannorm sample_emb embz edt band sheet ct_augment scan_augment
         spatial_augment crush scrollaug cover sources band_onfly fuse fiberraster afv reg regmap zarr2 srpair view zarr3 store json z3w
         tiff zipr hf NN LP LIBS m curl zstd blosc z crypto CUDA)
ufsm_exe(test_phase_pool SUITE tests/test_phase_pool.c OBJS band_onfly fuse band edt zarr3 zarr2 store json z3w hf NN LP LIBS m curl zstd
         blosc z crypto CUDA)
ufsm_exe(test_phasepack SUITE tests/test_phasepack.c OBJS phasepack store zarr3 json z3w LIBS LDLIBS)
ufsm_exe(test_predict_det SUITE tests/test_predict_det.c OBJS unet json zarr3 store z3w NN LP LIBS LDLIBS CUDA DEPS ufsm)
ufsm_exe(test_predict_feather SUITE tests/test_predict_feather.c OBJS unet json zarr3 store z3w NN LP LIBS LDLIBS CUDA DEPS ufsm)
ufsm_exe(test_predict_fitgrid SUITE tests/test_predict_fitgrid.c OBJS unet fit fit_emb embz fit_ct l1pot coreline json zarr3 store z3w NN LP
         LIBS LDLIBS CUDA DEPS ufsm)
ufsm_exe(test_predict_unwrap2 SUITE tests/test_predict_unwrap2.c OBJS unet fit fit_emb embz fit_ct l1pot coreline json zarr3 store z3w NN LP
         LIBS LDLIBS CUDA DEPS ufsm)
ufsm_exe(test_pyports SUITE tests/test_pyports.c OBJS coreline_check coreline imgops jpegpar tiff tiffw fit fit_emb embz z3w fit_ct l1pot
         json zarr3 store NN LP LIBS LDLIBS CUDA)
ufsm_exe(test_qkclip SUITE tests/test_qkclip.c OBJS NN LP LIBS LDLIBS CUDA)
ufsm_exe(test_raster SUITE tests/test_raster.c OBJS scannorm edt band sheet ct_augment scan_augment spatial_augment crush scrollaug cover
         json store zarr3 sources band_onfly fuse fiberraster afv regmap srpair zarr2 tiff z3w hf zipr view reg sample_emb embz surfcomp
         LIBS LDLIBS)
ufsm_exe(test_rc SUITE tests/test_rc.c OBJS NN LP LIBS m CUDA)
ufsm_exe(test_recompute_live SUITE tests/test_recompute_live.c OBJS unet NN LP LIBS m CUDA)
ufsm_exe(test_recto SUITE tests/test_recto.c OBJS rectoruler rectocls rectosp reg fit fit_emb embz z3w fit_ct l1pot coreline json zarr3
         zarr2 store NN LP LIBS LDLIBS CUDA)
ufsm_exe(test_reg SUITE tests/test_reg.c OBJS reg reg_init zarr3 zarr2 store json LIBS LDLIBS)
ufsm_exe(test_reg_render SUITE tests/test_reg_render.c OBJS CORE NN LP LIBS LDLIBS CUDA)
ufsm_exe(test_regz2 SUITE tests/test_regz2.c OBJS reg zarr3 zarr2 store json LIBS LDLIBS)
ufsm_exe(test_render_fill SUITE tests/test_render_fill.c OBJS render inkcompare imgops jpegpar tiff tiffw z3w zarr3 store json LIBS LDLIBS)
ufsm_exe(test_render_fill_gpu SUITE tests/test_render_fill.c OBJS render inkcompare imgops jpegpar tiff tiffw z3w zarr3 store json nn_render
         nn_gpudec LIBS LDLIBS CUDA DEFS RENDER_GPU)
ufsm_exe(test_render_known SUITE tests/test_render_known.c OBJS render inkcompare imgops jpegpar tiff tiffw z3w zarr3 store json LIBS
         LDLIBS)
ufsm_exe(test_render_stream SUITE tests/test_render_stream.c OBJS render inkcompare imgops jpegpar tiff tiffw z3w zarr3 store json LIBS
         LDLIBS)
ufsm_exe(test_render_stream_gpu SUITE tests/test_render_stream.c OBJS render inkcompare imgops jpegpar tiff tiffw z3w zarr3 store json
         nn_render nn_gpudec LIBS LDLIBS CUDA DEFS RENDER_GPU)
ufsm_exe(test_repair SUITE tests/test_repair.c OBJS fit fit_emb embz z3w fit_ct l1pot coreline json zarr3 store NN LP LIBS LDLIBS CUDA)
ufsm_exe(test_replay_lean SUITE tests/test_replay_lean.c OBJS sample scannorm edt band sheet ct_augment scan_augment spatial_augment crush
         scrollaug cover json store zarr3 sources band_onfly fuse fiberraster afv regmap srpair zarr2 tiff z3w hf zipr view reg sample_emb
         embz NN LP LIBS LDLIBS CUDA)
ufsm_exe(test_restore_task SUITE tests/test_restore_task.c OBJS CORE NN LP LIBS LDLIBS CUDA)
ufsm_exe(test_robust SUITE tests/test_robust.c OBJS packmap z3w zarr3 store json NN LP LIBS LDLIBS CUDA)
ufsm_exe(test_rungs SUITE tests/test_rungs.c OBJS sample scannorm sample_emb embz edt band sheet ct_augment scan_augment spatial_augment
         crush scrollaug cover sources band_onfly fuse fiberraster afv reg regmap zarr2 srpair view zarr3 store json z3w tiff zipr hf NN LP
         LIBS m curl zstd blosc z crypto CUDA)
ufsm_exe(test_rv SUITE tests/test_rv.c OBJS NN LP LIBS LDLIBS CUDA DEPS ufsm make_wind_fixture)
ufsm_exe(test_sample_ops SUITE tests/test_sample_ops.c OBJS scannorm edt band sheet ct_augment scan_augment spatial_augment crush scrollaug
         cover json store zarr3 sources band_onfly fuse fiberraster afv regmap srpair zarr2 tiff z3w hf zipr view reg sample_emb embz LIBS
         LDLIBS)
ufsm_exe(test_sampler_safety SUITE tests/test_sampler_safety.c OBJS scannorm edt band sheet ct_augment scan_augment spatial_augment crush
         scrollaug cover json store zarr3 sources band_onfly fuse fiberraster afv regmap srpair zarr2 tiff z3w hf zipr view reg sample_emb
         embz LIBS LDLIBS)
ufsm_exe(test_scan_augment SUITE tests/test_scan_augment.c OBJS ct_augment LIBS m)
ufsm_exe(test_scanprep SUITE tests/test_scanprep.c OBJS scannorm edt band sheet ct_augment scan_augment spatial_augment crush scrollaug
         cover json store zarr3 sources band_onfly fuse fiberraster afv regmap srpair zarr2 tiff z3w hf zipr view reg sample_emb embz
         scanprep LIBS LDLIBS)
ufsm_exe(test_score_par SUITE tests/test_score_par.c OBJS z3w zarr3 store json LIBS LDLIBS DEPS ufsm)
ufsm_exe(test_scrollaug SUITE tests/test_scrollaug.c OBJS scannorm edt band sheet ct_augment scan_augment spatial_augment crush scrollaug
         cover json store zarr3 sources band_onfly fuse fiberraster afv regmap srpair zarr2 tiff z3w hf zipr view reg sample_emb embz LIBS
         LDLIBS)
ufsm_exe(test_share_enc_a1 SUITE tests/test_share_enc_a1.c OBJS unet NN LP LIBS m CUDA)
ufsm_exe(test_sheet SUITE tests/test_sheet.c OBJS sheet json LIBS crypto m)
ufsm_exe(test_sheet_gpu SUITE tests/test_sheet_gpu.c OBJS split NN LP LIBS m CUDA)
ufsm_exe(test_sheetcut SUITE tests/test_sheetcut.c OBJS sheetcut tfx imgops jpegpar tiffw tiff surfcomp LIBS LDLIBS)
ufsm_exe(test_sheetpbr SUITE tests/test_sheetpbr.c OBJS sheetpbr reg zarr2 imgops jpegpar json zarr3 store z3w tiffw tiff tfx surfcomp LIBS
         LDLIBS)
ufsm_exe(test_sheetscore SUITE tests/test_sheetscore.c OBJS sheetscore scannorm edt band sheet ct_augment scan_augment spatial_augment crush
         scrollaug cover json store zarr3 sources band_onfly fuse fiberraster afv regmap srpair zarr2 tiff z3w hf zipr view reg sample_emb
         embz NN LP LIBS LDLIBS CUDA)
ufsm_exe(test_side SUITE tests/test_side.c OBJS band LIBS m)
ufsm_exe(test_spatial_augment SUITE tests/test_spatial_augment.c OBJS scannorm edt band sheet ct_augment scan_augment spatial_augment crush
         scrollaug cover json store zarr3 sources band_onfly fuse fiberraster afv regmap srpair zarr2 tiff z3w hf zipr view reg sample_emb
         embz LIBS LDLIBS)
ufsm_exe(test_spiralfit SUITE tests/test_spiralfit.c OBJS countfield ctcal fit fit_emb embz fit_ct l1pot coreline json zarr3 store z3w NN LP
         LIBS LDLIBS CUDA)
ufsm_exe(test_split SUITE tests/test_split.c OBJS unet split NN LP LIBS m CUDA)
ufsm_exe(test_split_gnbwd SUITE tests/test_split_gnbwd.c OBJS split NN LP LIBS m CUDA)
ufsm_exe(test_split_lat SUITE tests/test_split_lat.c OBJS unet split NN LP LIBS m CUDA)
ufsm_exe(test_ssl SUITE tests/test_ssl.c OBJS NN LP LIBS LDLIBS CUDA)
ufsm_exe(test_sslstat SUITE tests/test_sslstat.c LIBS m DEPS)
ufsm_exe(test_stem_precision SUITE tests/test_stem_precision.c OBJS NN LP LIBS m CUDA)
ufsm_exe(test_stitch SUITE tests/test_stitch.c OBJS stitch fit fit_emb embz fit_ct l1pot coreline json zarr3 store z3w tiffw NN LP LIBS
         LDLIBS CUDA)
ufsm_exe(test_strip SUITE tests/test_strip.c OBJS strip flatten imgops jpegpar tiffw tiff tfx surfcomp LIBS LDLIBS)
ufsm_exe(test_strip_gpu SUITE tests/test_strip.c OBJS strip flatten imgops jpegpar tiffw tiff tfx surfcomp nn_flat LIBS LDLIBS CUDA DEFS
         STRIP_GPU)
ufsm_exe(test_stripdistort SUITE tests/test_stripdistort.c OBJS stripdistort imgops jpegpar tiffw tiff LIBS LDLIBS)
ufsm_exe(test_surf SUITE tests/test_surf.c OBJS surf stitch z3w fit fit_emb embz fit_ct l1pot coreline tiffw tiff json zarr3 store NN LP
         imgops jpegpar LIBS LDLIBS CUDA)
ufsm_exe(test_surflabels SUITE tests/test_surflabels.c OBJS surflabels phasecoh pairacc tfx surfcomp tiffw scannorm edt band sheet
         ct_augment scan_augment spatial_augment crush scrollaug cover json store zarr3 sources band_onfly fuse fiberraster afv regmap
         srpair zarr2 tiff z3w hf zipr view reg sample_emb embz NN LP LIBS LDLIBS CUDA)
ufsm_exe(test_sweep SUITE tests/test_sweep.c OBJS CORE NN LP LIBS LDLIBS CUDA)
ufsm_exe(test_sysinfo SUITE tests/test_sysinfo.c DEPS)
ufsm_exe(test_target_erode SUITE tests/test_target_erode.c DEPS)
ufsm_exe(test_tfx SUITE tests/test_tfx.c OBJS tfx imgops jpegpar tiffw tiff surfcomp LIBS LDLIBS)
ufsm_exe(test_tiffw SUITE tests/test_tiffw.c OBJS tiffw tiff LIBS LDLIBS)
ufsm_exe(test_tta SUITE tests/test_tta.c OBJS NN LP LIBS LDLIBS CUDA)
ufsm_exe(test_typed_zarr SUITE tests/test_typed_zarr.c OBJS z3w zarr3 store json LIBS LDLIBS)
ufsm_exe(test_unet SUITE tests/test_unet.c OBJS unet NN LP LIBS m CUDA)
ufsm_exe(test_unwrap2_ink SUITE tests/test_unwrap2_ink.c OBJS unet json zarr3 store z3w NN LP LIBS LDLIBS CUDA DEPS ufsm make_wind_fixture)
ufsm_exe(test_unwrap2_sampler SUITE tests/test_unwrap2_sampler.c OBJS sample scannorm sample_emb embz edt band sheet ct_augment scan_augment
         spatial_augment crush scrollaug cover sources band_onfly fuse fiberraster afv reg regmap zarr2 srpair view zarr3 store json z3w
         tiff zipr hf NN LP LIBS m curl zstd blosc z crypto CUDA)
ufsm_exe(test_unwrap_targets SUITE tests/test_unwrap_targets.c OBJS NN LP LIBS m CUDA)
ufsm_exe(test_vcgpu SUITE tests/test_vcgpu.c OBJS vcgpu NN LP LIBS LDLIBS CUDA INC CUDA)
ufsm_exe(test_volcomp SUITE tests/test_volcomp.c LIBS m)
ufsm_exe(test_volcomp_fast SUITE tests/test_volcomp_fast.c OBJS zarr3 store json z3w LIBS LDLIBS)
ufsm_exe(test_votes SUITE tests/test_votes.c OBJS votes scannorm edt band sheet ct_augment scan_augment spatial_augment crush scrollaug
         cover json store zarr3 sources band_onfly fuse fiberraster afv regmap srpair zarr2 tiff z3w hf zipr view reg sample_emb embz LIBS
         LDLIBS)
ufsm_exe(test_vwi SUITE tests/test_vwi.c OBJS NN LP LIBS LDLIBS CUDA)
ufsm_exe(test_vwt SUITE tests/test_vwt.c OBJS vwt_net NN LP LIBS LDLIBS CUDA)
ufsm_exe(test_vwt_aug SUITE tests/test_vwt_aug.c OBJS vwt_aug deform scannorm edt band sheet ct_augment scan_augment spatial_augment crush
         scrollaug cover json store zarr3 sources band_onfly fuse fiberraster afv regmap srpair zarr2 tiff z3w hf zipr view reg sample_emb
         embz LIBS LDLIBS)
ufsm_exe(test_vwt_fast SUITE tests/test_vwt_fast.c OBJS NN LP LIBS LDLIBS CUDA)
ufsm_exe(test_vwt_mv SUITE tests/test_vwt_mv.c OBJS vwt_mv NN LP LIBS LDLIBS CUDA)
ufsm_exe(test_vwt_sample SUITE tests/test_vwt_sample.c OBJS vwt_sample imgops jpegpar tiffw tfx surfcomp vwt_aug zcache deform edt band
         sheet ct_augment scan_augment spatial_augment crush scrollaug cover json store zarr3 sources band_onfly fuse fiberraster afv reg
         regmap zarr2 srpair tiff z3w hf zipr view sample_emb embz NN LP LIBS LDLIBS CUDA)
ufsm_exe(test_w4dev SUITE tests/test_w4dev.c OBJS band NN LP LIBS m CUDA)
ufsm_exe(test_wgrad_grid SUITE tests/test_wgrad_grid.c OBJS NN LP LIBS m CUDA)
ufsm_exe(test_wgrad_staging SUITE tests/test_wgrad_staging.c OBJS NN LP LIBS m CUDA)
ufsm_exe(test_wide_up_grad SUITE tests/test_wide_up_grad.c OBJS NN LP LIBS m CUDA)
ufsm_exe(test_winauto SUITE tests/test_winauto.c DEPS)
ufsm_exe(test_wind SUITE tests/test_wind.c OBJS wind LIBS m)
ufsm_exe(test_wind4 SUITE tests/test_wind4.c OBJS band NN LP LIBS m CUDA)
ufsm_exe(test_wind_sampler SUITE tests/test_wind_sampler.c OBJS scannorm edt band sheet ct_augment scan_augment spatial_augment crush
         scrollaug cover json store zarr3 sources band_onfly fuse fiberraster afv regmap srpair zarr2 tiff z3w hf zipr view reg sample_emb
         embz LIBS LDLIBS)
ufsm_exe(test_wrap_check SUITE tests/test_wrap_check.c OBJS wrapcheck fit fit_emb embz z3w fit_ct l1pot coreline json zarr3 zarr2 store NN
         LP LIBS LDLIBS CUDA)
ufsm_exe(test_wraprepair SUITE tests/test_wraprepair.c OBJS wraprepair tfx surfcomp tiffw scannorm edt band sheet ct_augment scan_augment
         spatial_augment crush scrollaug cover json store zarr3 sources band_onfly fuse fiberraster afv regmap srpair zarr2 tiff z3w hf zipr
         view reg sample_emb embz NN LP LIBS LDLIBS CUDA)
ufsm_exe(test_wraps_core SUITE tests/test_wraps_core.c OBJS fit fit_emb embz z3w fit_ct l1pot coreline tiff json zarr3 store NN LP LIBS
         LDLIBS CUDA DEPS ufsm)
ufsm_exe(test_wrapsnap SUITE tests/test_wrapsnap.c OBJS wrapsnap tfx surfcomp tiffw scannorm edt band sheet ct_augment scan_augment
         spatial_augment crush scrollaug cover json store zarr3 sources band_onfly fuse fiberraster afv regmap srpair zarr2 tiff z3w hf zipr
         view reg sample_emb embz NN LP LIBS LDLIBS CUDA)
ufsm_exe(test_wtrunk SUITE tests/test_wtrunk.c OBJS wtrunk unet checkpoint json NN LP LIBS LDLIBS CUDA)
ufsm_exe(test_z3_autosub SUITE tests/test_z3_autosub.c OBJS z3w zarr3 store json LIBS LDLIBS)
ufsm_exe(test_z3_sink SUITE tests/test_z3_sink.c OBJS z3w zarr3 store json LIBS LDLIBS)
ufsm_exe(test_zcache SUITE tests/test_zcache.c OBJS zcache dcache zarr3 store json z3w hf zipr LIBS LDLIBS)
ufsm_exe(test_zwin SUITE tests/test_zwin.c OBJS NN LP LIBS m CUDA)
ufsm_exe(train_lp tests/train_lp.c OBJS unet NN LP LIBS m CUDA)
ufsm_exe(xscan_render tests/xscan_render.c OBJS scannorm edt band sheet ct_augment scan_augment spatial_augment crush scrollaug cover json
         store zarr3 sources band_onfly fuse fiberraster afv regmap srpair zarr2 tiff z3w hf zipr view reg sample_emb embz imgops jpegpar
         tiffw tfx surfcomp LIBS LDLIBS)

# ported from the Makefile of master (7ee48e2 .. e478a9d) when the CMake build merged (2026-11-13)
ufsm_exe(bench_s2b tests/bench_s2b.c OBJS NN LP LIBS m CUDA)
ufsm_exe(bench_s2f tests/bench_s2f.c OBJS NN LP LIBS m CUDA)
ufsm_exe(test_head_tc SUITE tests/test_head_tc.c OBJS NN LP LIBS m CUDA)
ufsm_exe(test_skip_empty SUITE tests/test_skip_empty.c OBJS unet NN LP LIBS m CUDA)
ufsm_exe(test_grad_ar SUITE tests/test_grad_ar.c OBJS CORE NN LP LIBS LDLIBS CUDA)
ufsm_exe(test_dcache_batch SUITE tests/test_dcache_batch.c OBJS dcache zarr3 store z3w json LIBS LDLIBS)
ufsm_exe(bench_fstem tests/bench_fstem.c OBJS CORE NN LP LIBS LDLIBS CUDA)
ufsm_exe(bench_wgl0 tests/bench_wgl0.c OBJS CORE NN LP LIBS LDLIBS CUDA)
# 2:4 sparse MX-fp4 convolutions (src/lp_sp24*.cu, DESIGN.md 2026-11-13): instruction layout + throughput, conv kernels vs host
ufsm_exe(test_sp24_mma SUITE tests/test_sp24_mma.c OBJS NN LP LIBS m CUDA)
ufsm_exe(test_sp24_conv SUITE tests/test_sp24_conv.c OBJS NN LP LIBS m CUDA)
ufsm_exe(bench_sp24 tests/bench_sp24.c OBJS NN LP LIBS m CUDA)
ufsm_exe(s2b_replay tests/s2b_replay.c OBJS NN LP LIBS m CUDA)
ufsm_exe(test_widen SUITE tests/test_widen.c OBJS unet NN LP LIBS m CUDA)
