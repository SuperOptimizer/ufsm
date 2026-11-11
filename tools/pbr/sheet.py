# sheet.py OUTDIR NAME: PNG copies of the pgm / ppm images of a sheet-pbr run and a labelled contact sheet (ink | flat CT | maps | shading)
import sys, os, numpy as np
from PIL import Image, ImageDraw
d, nm = sys.argv[1], sys.argv[2]; Image.MAX_IMAGE_PIXELS = None
os.makedirs(d + '/png', exist_ok=True)
for f in sorted(os.listdir(d)):
    if f.endswith('.pgm') or f.endswith('.ppm'): Image.open(d + '/' + f).save(d + '/png/' + f[:-4] + '.png')
panels = [('published ink PREDICTION (model output, not a label)', nm + '_ink.png'), ('flat CT at the mesh', 'png/flat.png'), ('max CT +-3 (ufsm render)', 'png/flatmax.png'),
          ('near-recto CT (first 4.8 um)', 'png/near_recto.png'), ('recto height', 'png/height.png'), ('height high-pass', 'png/highpass.png'),
          ('roughness', 'png/roughness.png'), ('thickness (recto -> verso)', 'png/thickness.png'), ('raking light 135 deg', 'png/light_04.png'),
          ('RTI min over 12 lights', 'png/rti_min.png'), ('PBR (GGX, shadows, AO)', 'png/pbr.png'), ('PBR, albedo = near-recto CT', 'png/pbr_density.png')]
S = 512; im = Image.new('RGB', (4 * S, 3 * (S + 24)), 'white'); dr = ImageDraw.Draw(im)
for k, (t, f) in enumerate(panels):
    p = os.path.join(os.path.dirname(d), f) if f.endswith('_ink.png') else os.path.join(d, f)
    if not os.path.exists(p): continue
    a = Image.open(p).convert('RGB').resize((S, S), Image.LANCZOS); x, y = (k % 4) * S, (k // 4) * (S + 24)
    im.paste(a, (x, y + 24)); dr.text((x + 6, y + 6), t, fill='black')
im.save(d + '/contact.png'); print(d + '/contact.png')
