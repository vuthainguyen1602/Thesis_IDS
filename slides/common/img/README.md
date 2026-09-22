# Logo assets

`logouit.png` is a slides-specific derivative of `thesis/img/logouit.png`.
Two things make the original unusable in these decks:

1. It is a **16-bit-per-channel** PNG. XeLaTeX's `xdvipdfmx` driver reserves
   the box but draws nothing — the image silently disappears. (pdflatex
   renders it fine, which is why the thesis cover page still works.)
2. Its "transparent" backdrop is actually **opaque white** baked into the
   pixels, so it shows as a white rectangle against metropolis' `#FAFAFA`
   page.

Regenerate after any change to the source logo:

```python
from PIL import Image
src = Image.open('thesis/img/logouit.png').convert('RGBA')
BG  = (250, 250, 250)                      # metropolis page colour, black!2
out = Image.new('RGB', src.size)
sp, op = src.convert('RGB').load(), out.load()
for y in range(src.size[1]):
    for x in range(src.size[0]):
        r, g, b = sp[x, y]
        op[x, y] = BG if (r > 244 and g > 244 and b > 244) else (r, g, b)
out.save('slides/common/img/logouit.png', optimize=True)
```
