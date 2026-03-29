from PIL import Image
import numpy as np

INPUT_IMAGE  = "input.jpg"
OUTPUT_IMAGE = "output.jpg"

# --- Preprocessing controls ---
# Brightness: positive values lighten, negative values darken (-255 to 255)
BRIGHTNESS = 30
# Contrast: >1.0 increases separation, <1.0 flattens tones (0.5 to 2.0)
CONTRAST = 1.2
# Gamma: <1.0 lifts midtones, >1.0 darkens midtones (0.4 to 2.5)
GAMMA = 0.9
# Scale: target width in pixels — None keeps original size
SCALE = None


def preprocess(gray, brightness, contrast, gamma, scale):
    h, w = gray.shape

    # Resize if a target width is given
    if scale is not None:
        new_w = int(scale)
        new_h = int(h * new_w / w)
        pil_img = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8))
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
        gray = np.array(pil_img, dtype=float)

    # Brightness: shift all values
    gray = gray + brightness

    # Contrast: scale around midpoint 128
    gray = (gray - 128) * contrast + 128

    # Gamma: normalize to [0,1], apply curve, restore to [0,255]
    gray = np.clip(gray, 0, 255)
    gray = 255.0 * (gray / 255.0) ** gamma

    # Final clamp to valid range
    gray = np.clip(gray, 0, 255)

    return gray


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
            buf[current][x + 2]     += error * 7 / 16
            buf[1 - current][x]     += error * 3 / 16
            buf[1 - current][x + 1] += error * 5 / 16
            buf[1 - current][x + 2] += error * 1 / 16

        # Clear the processed row and rotate to the next
        buf[current][:] = 0
        current = 1 - current

    return out


img  = Image.open(INPUT_IMAGE).convert("L")
gray = np.array(img, dtype=float)

gray = preprocess(gray, BRIGHTNESS, CONTRAST, GAMMA, SCALE)
out  = floyd_steinberg(gray)
Image.fromarray(out).save(OUTPUT_IMAGE)
print("done")
