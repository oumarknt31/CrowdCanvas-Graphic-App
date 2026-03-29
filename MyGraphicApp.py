from PIL import Image
import numpy as np

INPUT_IMAGE = "input.jpg"
OUTPUT_IMAGE = "output.jpg"


def floyd_steinberg(gray):
    h, w = gray.shape
    out = np.full((h, w), 255, dtype=np.uint8)

    # Two padded row buffers: each holds w + 2 values (1 padding cell on each side)
    buf = [np.zeros(w + 2, dtype=float), np.zeros(w + 2, dtype=float)]
    current = 0

    for y in range(h):
        # Add source gray values into the current buffer on top of any accumulated error
        buf[current][1:w + 1] += gray[y]

        for x in range(w):
            val = buf[current][x + 1]
            new = 0 if val < 128 else 255
            error = val - new
            out[y, x] = new

            # Spread error — no boundary checks needed, padding absorbs edge writes
            buf[current][x + 2]       += error * 7 / 16
            buf[1 - current][x]       += error * 3 / 16
            buf[1 - current][x + 1]   += error * 5 / 16
            buf[1 - current][x + 2]   += error * 1 / 16

        # Clear the processed row and rotate to the next
        buf[current][:] = 0
        current = 1 - current

    return out


img = Image.open(INPUT_IMAGE).convert("L")
gray = np.array(img, dtype=float)

out = floyd_steinberg(gray)
Image.fromarray(out).save(OUTPUT_IMAGE)
print("done")
