"""Render a terminal-style PNG from selected lines of a board log (text kept verbatim)."""
import sys, re
from PIL import Image, ImageDraw, ImageFont
log, out, title, spec = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
raw = open(log, encoding="utf-8", errors="replace", newline="").read().split("\n")
# A terminal shows only the text after the last carriage return on each line.
lines = [l.split("\r")[-1] for l in raw]
lines = [re.sub(r"\x1b\[[0-9;]*m", "", l) for l in lines]
sel = []
for part in spec.split(","):
    if part == "gap":
        sel.append(None); continue
    a, b = map(int, part.split("-"))
    sel += [l for l in lines[a-1:b] if not l.lstrip().startswith("[Stage ")]
font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 26)
lh, pad, bar = 34, 24, 56
width = max(font.getlength(l or "") for l in sel if l is not None) + 2*pad
img = Image.new("RGB", (int(max(width, 1600)), bar + pad + lh*len(sel) + pad), (255, 255, 255))
d = ImageDraw.Draw(img)
d.rectangle([0, 0, img.width, bar], fill=(232, 232, 232)); d.line([0, bar, img.width, bar], fill=(200, 200, 200), width=2)
for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
    d.ellipse([20 + i*34, 18, 40 + i*34, 38], fill=c)
tf = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 24)
d.text(((img.width - tf.getlength(title))/2, 14), title, fill=(60, 60, 60), font=tf)
y = bar + pad
for l in sel:
    if l is None:
        d.text((pad, y), "   [...]", fill=(150, 150, 150), font=font)
    else:
        col = (30, 30, 30)
        if "[OK]" in l: col = (0, 120, 60)
        elif "[WARN]" in l: col = (170, 110, 0)
        elif "MONITOR" in l or "Temp:" in l: col = (20, 70, 150)
        d.text((pad, y), l, fill=col, font=font)
    y += lh
img.save(out)
print(out, img.size)
