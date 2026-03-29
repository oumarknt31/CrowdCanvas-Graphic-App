from PIL import Image, ImageDraw
import numpy as np

INPUT_IMAGE = "input.jpg"
OUTPUT_IMAGE = "output.jpg"

img = Image.open(INPUT_IMAGE).convert("L")
gray = np.array(img, dtype=float)

h, w = gray.shape

# create empty white board
canvas = Image.new("L", (w, h), 255)
draw = ImageDraw.Draw(canvas)

# Floyd–Steinberg
for y in range(h):
    for x in range(w):
        old = gray[y, x]

        if old < 128:
            new = 0
        else:
            new = 255

        if new == 0:
            draw.point((x, y), fill=0)

        error = old - new

        # spread  error
        if x + 1 < w:
            gray[y, x + 1] += error * 7 / 16
        if y + 1 < h:
            if x > 0:
                gray[y + 1, x - 1] += error * 3 / 16
            gray[y + 1, x] += error * 5 / 16
            if x + 1 < w:
                gray[y + 1, x + 1] += error * 1 / 16

canvas.save(OUTPUT_IMAGE)
print("done")