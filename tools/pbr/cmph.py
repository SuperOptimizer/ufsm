# cmph.py NAME: human-label comparison of one labelled crop at 2.4 um (out/h_s24_NAME) and 1.129 um (out/h_s113_NAME):
# row per resolution: HUMAN label, flat CT, near-recto CT, PBR, raking light, recto offset -> out/cmph_NAME.png
import sys, os, numpy as np
from PIL import Image, ImageDraw
nm = sys.argv[1]; S = 360
cols = [('HUMAN ink label', 'human_ink'), ('flat CT at the face', 'flat'), ('near-recto CT', 'near_recto'), ('PBR', 'pbr'),
        ('raking light 135 deg', 'light_04'), ('recto offset (relief)', 'recto_off')]
im = Image.new('RGB', (len(cols) * S, 2 * (S + 24) + 24), 'white'); dr = ImageDraw.Draw(im)
for r, (lab, d) in enumerate([('2.4 um', 'h_s24_' + nm), ('1.129 um (registered)', 'h_s113_' + nm)]):
    for c, (t, f) in enumerate(cols):
        p = 'out/%s/%s.pgm' % (d, f)
        if not os.path.exists(p): continue
        a = Image.open(p).convert('L').resize((S, S), Image.LANCZOS); x, y = c * S, 24 + r * (S + 24)
        im.paste(a.convert('RGB'), (x, y + 24)); dr.text((x + 6, y + 6), t + ('  |  ' + lab if c == 0 else ''), fill='black')
dr.text((6, 6), 'Paris 4 %s: human labels (ink/phercparis4) vs label-free surface maps, same 2.46 x 2.46 mm crop' % nm, fill='black')
im.save('out/cmph_%s.png' % nm); print('out/cmph_%s.png' % nm)
