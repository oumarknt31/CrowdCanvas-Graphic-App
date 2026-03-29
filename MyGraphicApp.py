import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np

PREVIEW_SIZE = 350  # pixel dimensions of each preview panel


def preprocess(gray, brightness, contrast, gamma, scale):
    h, w = gray.shape

    if scale is not None:
        new_w = int(scale)
        new_h = int(h * new_w / w)
        pil_img = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8))
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
        gray = np.array(pil_img, dtype=float)

    gray = gray + brightness
    gray = (gray - 128) * contrast + 128
    gray = np.clip(gray, 0, 255)
    gray = 255.0 * (gray / 255.0) ** gamma
    return np.clip(gray, 0, 255)


def floyd_steinberg(gray):
    h, w = gray.shape
    out = np.full((h, w), 255, dtype=np.uint8)
    buf = [np.zeros(w + 2, dtype=float), np.zeros(w + 2, dtype=float)]
    current = 0

    for y in range(h):
        buf[current][1:w + 1] += gray[y]
        for x in range(w):
            val = buf[current][x + 1]
            new = 0 if val < 128 else 255
            error = val - new
            out[y, x] = new
            buf[current][x + 2]     += error * 7 / 16
            buf[1 - current][x]     += error * 3 / 16
            buf[1 - current][x + 1] += error * 5 / 16
            buf[1 - current][x + 2] += error * 1 / 16
        buf[current][:] = 0
        current = 1 - current

    return out


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("CrowdCanvas Graphic App")
        self.root.resizable(False, False)

        self.original_gray = None  # full-resolution grayscale float array
        self.preview_gray  = None  # downscaled thumbnail for live preview
        self._pending      = None  # debounce handle

        self._build_ui()

    def _build_ui(self):
        # ── Row 0: two preview panels ──────────────────────────────────────
        top = tk.Frame(self.root, bg="#1a1a1a", padx=12, pady=12)
        top.grid(row=0, column=0, sticky="ew")

        self.lbl_input  = self._preview_panel(top, "Adjusted Input",   col=0)
        self.lbl_output = self._preview_panel(top, "Halftoned Output", col=1)

        # ── Row 1: controls ────────────────────────────────────────────────
        bottom = tk.Frame(self.root, padx=16, pady=12)
        bottom.grid(row=1, column=0, sticky="ew")

        # Load / Save / status bar
        bar = tk.Frame(bottom)
        bar.grid(row=0, column=0, columnspan=4, sticky="ew", pady=(0, 10))
        tk.Button(bar, text="Load Image",  width=14, command=self._load).pack(side="left", padx=(0, 8))
        tk.Button(bar, text="Save Output", width=14, command=self._save).pack(side="left")
        self.status = tk.Label(bar, text="Load an image to begin.", fg="gray", anchor="w")
        self.status.pack(side="left", padx=(16, 0))

        # Sliders
        self.brightness = self._slider(bottom, "Brightness", -255, 255,  30,  col=0)
        self.contrast   = self._slider(bottom, "Contrast",   0.5,  2.0,  1.2, col=1, is_float=True)
        self.gamma      = self._slider(bottom, "Gamma",      0.4,  2.5,  0.9, col=2, is_float=True)

        # Resolution column
        res = tk.Frame(bottom)
        res.grid(row=1, column=3, padx=(16, 0), sticky="nw")
        self.use_scale = tk.BooleanVar(value=False)
        tk.Checkbutton(res, text="Custom width (px)",
                       variable=self.use_scale,
                       command=self._on_change).pack(anchor="w")
        self.val_scale = tk.Label(res, text="800 px", width=8, anchor="e")
        self.val_scale.pack(anchor="e")
        self.scale_slider = tk.Scale(res, from_=100, to=1500,
                                     orient="horizontal", length=160,
                                     showvalue=False, command=self._on_scale_move)
        self.scale_slider.set(800)
        self.scale_slider.pack()

    def _preview_panel(self, parent, title, col):
        frame = tk.Frame(parent, bg="#1a1a1a")
        frame.grid(row=0, column=col, padx=8)
        tk.Label(frame, text=title, bg="#1a1a1a", fg="#aaa",
                 font=("Helvetica", 10)).pack(pady=(0, 4))
        box = tk.Frame(frame, bg="#333", width=PREVIEW_SIZE, height=PREVIEW_SIZE)
        box.pack_propagate(False)
        box.pack()
        lbl = tk.Label(box, bg="#333", text="—", fg="#555")
        lbl.place(relx=0.5, rely=0.5, anchor="center")
        return lbl

    def _slider(self, parent, label, from_, to, default, col, is_float=False):
        frame = tk.Frame(parent)
        frame.grid(row=1, column=col, padx=(0, 8), sticky="nw")

        header = tk.Frame(frame)
        header.pack(fill="x")
        tk.Label(header, text=label, anchor="w").pack(side="left")
        val_lbl = tk.Label(header, anchor="e", width=6)
        val_lbl.pack(side="right")

        # Store floats as integers × 100 to avoid Tkinter float slider limitations
        factor = 100 if is_float else 1
        int_from    = round(from_   * factor)
        int_to      = round(to      * factor)
        int_default = round(default * factor)

        def on_move(v, lbl=val_lbl, f=is_float):
            display = float(v) / 100 if f else int(float(v))
            lbl.config(text=f"{display:.2f}" if f else str(display))
            self._on_change()

        s = tk.Scale(frame, from_=int_from, to=int_to,
                     orient="horizontal", length=160,
                     showvalue=False, command=on_move)
        s.set(int_default)
        s.pack()
        on_move(int_default)
        return s

    def _on_scale_move(self, v):
        self.val_scale.config(text=f"{int(float(v))} px")
        self._on_change()

    def _on_change(self, *_):
        """Debounce: wait 120 ms after the last slider move before updating."""
        if self.preview_gray is None:
            return
        if self._pending:
            self.root.after_cancel(self._pending)
        self._pending = self.root.after(120, self._update_previews)

    def _load(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if not path:
            return
        img = Image.open(path).convert("L")
        self.original_gray = np.array(img, dtype=float)

        # Downscale once for all live preview processing
        thumb = img.copy()
        thumb.thumbnail((PREVIEW_SIZE, PREVIEW_SIZE), Image.LANCZOS)
        self.preview_gray = np.array(thumb, dtype=float)

        self.status.config(text=f"Loaded: {path.split('/')[-1]}", fg="black")
        self._update_previews()

    def _params(self):
        brightness = self.brightness.get()
        contrast   = self.contrast.get()   / 100
        gamma      = self.gamma.get()      / 100
        scale      = self.scale_slider.get() if self.use_scale.get() else None
        return brightness, contrast, gamma, scale

    def _update_previews(self):
        """Run preprocessing + halftoning on the thumbnail and refresh both panels."""
        if self.preview_gray is None:
            return
        brightness, contrast, gamma, _ = self._params()

        # Live preview always uses the thumbnail — scale is ignored here
        adjusted  = preprocess(self.preview_gray, brightness, contrast, gamma, scale=None)
        halftoned = floyd_steinberg(adjusted)

        self._show(self.lbl_input,  Image.fromarray(np.clip(adjusted, 0, 255).astype(np.uint8)))
        self._show(self.lbl_output, Image.fromarray(halftoned))
        self.status.config(text="Live preview  ·  Save for full resolution.", fg="#555")

    def _show(self, label, img):
        img = img.copy()
        img.thumbnail((PREVIEW_SIZE, PREVIEW_SIZE), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(img)
        label.config(image=tk_img, text="")
        label.image = tk_img  # prevent garbage collection

    def _save(self):
        if self.original_gray is None:
            messagebox.showwarning("Nothing to Save", "Load an image first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")]
        )
        if not path:
            return
        self.status.config(text="Saving at full resolution...", fg="blue")
        self.root.update()
        brightness, contrast, gamma, scale = self._params()
        gray = preprocess(self.original_gray, brightness, contrast, gamma, scale)
        out  = floyd_steinberg(gray)
        Image.fromarray(out).save(path)
        self.status.config(text=f"Saved: {path.split('/')[-1]}", fg="green")


root = tk.Tk()
App(root)
root.mainloop()
