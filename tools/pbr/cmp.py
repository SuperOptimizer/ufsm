# cmp.py NAME: the same 1 x 1 mm patch (centre of the 1024-px sub-crop) from the 2.4 um scan (ss 2 sampling) and the 1.129 um scan:
# flat CT, near-recto CT, PBR, raking light 135 deg -> out/cmp_NAME.png (row 1: 2.4 um, row 2: 1.129 um)
import sys, os, numpy as np
from PIL import Image, ImageDraw
nm = sys.argv[1]; S = 520; a0, a1 = 1024 - 417, 1024 + 417   # 834 samples at ss 2 = 417 px of 2.4 um = 1.0 mm
cols = [('flat CT at the mesh', 'flat.png'), ('near-recto CT (first 4.8 um)', 'near_recto.png'), ('PBR (GGX, shadows, AO)', 'pbr.png'), ('raking light 135 deg', 'light_04.png')]
im = Image.new('RGB', (4 * S, 2 * (S + 24) + 24), 'white'); dr = ImageDraw.Draw(im)
for r, (lab, d) in enumerate([('2.4 um scan (sampled every 1.2 um)', 's24x2_' + nm), ('1.129 um scan (registered, sampled every 1.13 um)', 's113_' + nm)]):
    for c, (t, f) in enumerate(cols):
        p = 'out/%s/png/%s' % (d, f)
        if not os.path.exists(p): continue
        a = Image.open(p).convert('L').crop((a0, a0, a1, a1)).resize((S, S), Image.LANCZOS); x, y = c * S, 24 + r * (S + 24)
        im.paste(a.convert('RGB'), (x, y + 24)); dr.text((x + 6, y + 6), t + ('  |  ' + lab if c == 0 else ''), fill='black')
dr.text((6, 6), 'Paris 4 %s: the same 1 x 1 mm patch of the recto surface' % nm, fill='black'); im.save('out/cmp_%s.png' % nm); print('out/cmp_%s.png' % nm)
