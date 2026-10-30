"""Reference forward of the villa winding model (HF scrollprize/winding_model_9um, ckpt_final.pth) from the authors' source
(ScrollPrize/villa vesuvius/src/vesuvius/neural_tracing/winding_models/winding_model.py @37d1897, sparse clone under
UFSM_VILLAWIND/src/villa), fp32 on the CPU, for tests/test_villawind.py only (never part of a pipeline). Data dir UFSM_VILLAWIND
(default ~/ufsm-data/villawind). run_slab(image u8 [T,T,L], valid bool) returns the intermediates the C port dumps."""
import importlib.util, os, pathlib

DATA = pathlib.Path(os.environ.get("UFSM_VILLAWIND", "~/ufsm-data/villawind")).expanduser()
CKPT = DATA / "ckpt_final.pth"
SRC = DATA / "src/villa/vesuvius/src/vesuvius/neural_tracing/winding_models/winding_model.py"


def build():
    import torch
    spec = importlib.util.spec_from_file_location("villa_winding_model", SRC)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    net = mod.WindingModel(ck["config"]["model"])
    net.load_state_dict(ck["model"]); net.eval()
    return net


def run_slab(net, image, valid):
    import numpy as np, torch
    out = {}
    hooks = []
    def keep(name, chlast):
        def h(m, i, o):
            t = o.detach()
            out[name] = (t[0].permute(1, 2, 3, 0) if chlast else t[0]).float().numpy().copy()
        return h
    for s, st in enumerate(net.stages):
        hooks.append(st.register_forward_hook(keep(f"stage{s}", True)))
    hooks.append(net.trunk_in.register_forward_hook(keep("trunk_in", True)))
    hooks.append(net.stem_blocks[-1].register_forward_hook(keep("stem", True)))
    for k, b in enumerate(net.attention_blocks):
        hooks.append(b.register_forward_hook(keep(f"block{k}", False)))
    hooks.append(net.decoder.register_forward_hook(keep("decoder", True)))
    hooks.append(net.full_resolution_head[2].register_forward_hook(keep("head0", True)))
    hooks.append(net.full_resolution_head.register_forward_hook(keep("logits", False)))
    with torch.no_grad():
        r = net(torch.from_numpy(image.astype(np.float32))[None], torch.from_numpy(valid.astype(bool))[None])
    for h in hooks:
        h.remove()
    out["phase"] = r["phase"][0].numpy()
    return out
