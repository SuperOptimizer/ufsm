# sheeth.py OUTDIR TITLE [PRED.png]: contact sheet of a sheet-pbr --labels run: human label (ink white, no ink grey, unlabelled navy),
# the label surface's offset from the tracked recto face, the published prediction (if given) and the physical maps / shading
import sys, os, json, numpy as np
from PIL import Image, ImageDraw
d, title = sys.argv[1], sys.argv[2]; pred = sys.argv[3] if len(sys.argv) > 3 else None; Image.MAX_IMAGE_PIXELS = None
W = json.load(open(d + '/map.json'))['W']; os.makedirs(d + '/png', exist_ok=True)
for f in sorted(os.listdir(d)):
    if f.endswith('.pgm') or f.endswith('.ppm'): Image.open(d + '/' + f).save(d + '/png/' + f[:-4] + '.png')
h = np.fromfile(d + '/human_ink.f32', np.float32).reshape(W, W); rgb = np.zeros((W, W, 3), np.uint8); rgb[...] = (20, 30, 70)
rgb[h < 0.5] = (130, 130, 130); rgb[h > 0.5] = (255, 255, 255); Image.fromarray(rgb).save(d + '/png/human_label.png')
dr = np.fromfile(d + '/human_dr.f32', np.float32).reshape(W, W); c = np.clip((dr + 10) / 20, 0, 1); rgb = np.zeros((W, W, 3), np.uint8); rgb[...] = (20, 30, 70)
m = np.isfinite(dr); rgb[m, 0] = (255 * c[m]).astype(np.uint8); rgb[m, 2] = (255 * (1 - c[m])).astype(np.uint8); rgb[m, 1] = (255 * (1 - 2 * np.abs(c[m] - 0.5))).astype(np.uint8)
Image.fromarray(rgb).save(d + '/png/human_dr.png')
panels = [('HUMAN ink label (white ink, grey no ink, navy none)', 'png/human_label.png'), ('label surface - tracked recto (blue -10 .. red +10 vox)', 'png/human_dr.png'),
          ('published ink PREDICTION (model)', pred), ('flat CT at the mesh', 'png/flat.png'), ('max CT +-3 (ufsm render)', 'png/flatmax.png'),
          ('near-recto CT (first 4.8 um)', 'png/near_recto.png'), ('recto height', 'png/height.png'), ('thickness (recto -> verso)', 'png/thickness.png'),
          ('roughness', 'png/roughness.png'), ('raking light 135 deg', 'png/light_04.png'), ('RTI min over 12 lights', 'png/rti_min.png'), ('PBR (GGX, shadows, AO)', 'png/pbr.png')]
S = 512; im = Image.new('RGB', (4 * S, 3 * (S + 24) + 24), 'white'); D = ImageDraw.Draw(im); D.text((6, 6), title, fill='black')
for k, (t, f) in enumerate(panels):
    if not f: continue
    p = f if os.path.isabs(f) else os.path.join(d, f)
    if not os.path.exists(p): continue
    a = Image.open(p).convert('RGB').resize((S, S), Image.LANCZOS); x, y = (k % 4) * S, 24 + (k // 4) * (S + 24)
    im.paste(a, (x, y + 24)); D.text((x + 6, y + 6), t, fill='black')
im.save(d + '/contact_human.png'); print(d + '/contact_human.png')
