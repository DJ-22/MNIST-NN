import tkinter as tk
import numpy as np
from PIL import Image, ImageFilter
from model import loadParam, forwProp


GRID = 28
SCALE = 20
SIZE = GRID * SCALE
DELAY = 120

BG = "#f5f7fa"
CARD = "#ffffff"
ACCENT = "#2563eb"
SOFT = "#93c5fd"
TEXT = "#1f2937"
SUBTEXT = "#6b7280"

class DigitUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("MNIST Digit Recognizer")
        self.root.configure(bg=BG)

        self.show_heatmap = False

        left = tk.Frame(self.root, bg=CARD, padx=14, pady=14)
        left.grid(row=0, column=0, rowspan=2, padx=12, pady=12)

        tk.Label(left, text="Draw a digit", bg=CARD, fg=TEXT,
                 font=("Segoe UI", 14, "bold")).pack(anchor="w", pady=(0, 6))

        self.canvas = tk.Canvas(
            left, width=SIZE, height=SIZE, bg="#0f172a",
            highlightthickness=2, highlightbackground="#e5e7eb"
        )
        self.canvas.pack()

        right = tk.Frame(self.root, bg=CARD, padx=14, pady=14)
        right.grid(row=0, column=1, padx=12, pady=12, sticky="n")

        tk.Label(right, text="Model confidence", bg=CARD, fg=TEXT,
                 font=("Segoe UI", 14, "bold")).pack(anchor="w", pady=(0, 6))

        self.bar_canvas = tk.Canvas(
            right, width=300, height=280, bg=CARD, highlightthickness=0
        )
        self.bar_canvas.pack()

        self.label = tk.Label(
            self.root, text="Prediction: —",
            font=("Segoe UI", 20, "bold"),
            fg=ACCENT, bg=BG
        )
        self.label.grid(row=1, column=1, pady=(0, 10))

        btn = tk.Frame(self.root, bg=BG)
        btn.grid(row=2, column=0, columnspan=2, pady=8)

        tk.Button(
            btn, text="Toggle Heatmap", command=self.toggleView,
            bg=ACCENT, fg="white", relief="flat",
            font=("Segoe UI", 11), padx=14, pady=6
        ).pack(side=tk.LEFT, padx=6)

        tk.Button(
            btn, text="Clear", command=self.clear,
            bg="#ef4444", fg="white", relief="flat",
            font=("Segoe UI", 11), padx=14, pady=6
        ).pack(side=tk.LEFT, padx=6)

        self.img = Image.new("L", (GRID, GRID), 0)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)

        self.pred_job = None

        param = loadParam()
        if not param:
            raise RuntimeError("Train the model first.")
        self.w1, self.b1, self.w2, self.b2 = param

        self.initBar()
        self.root.mainloop()

    def toggleView(self):
        self.show_heatmap = not self.show_heatmap
        self.redraw()

    def initBar(self):
        self.bars = []
        self.texts = []
        
        for i in range(10):
            y = 25 + i * 25
            
            self.bar_canvas.create_text(
                10, y, text=str(i), anchor="w",
                font=("Segoe UI", 10), fill=SUBTEXT
            )
            
            bar = self.bar_canvas.create_rectangle(
                35, y-7, 35, y+7, fill=SOFT, outline=""
            )
            txt = self.bar_canvas.create_text(
                290, y, text="0.0%", anchor="e",
                font=("Segoe UI", 10), fill=SUBTEXT
            )
            
            self.bars.append(bar)
            self.texts.append(txt)

    def paint(self, e):
        x = e.x // SCALE
        y = e.y // SCALE

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < GRID and 0 <= ny < GRID:
                    val = self.img.getpixel((nx, ny))
                    self.img.putpixel((nx, ny), min(255, val + 70))

        self.redraw()

        if self.pred_job:
            self.root.after_cancel(self.pred_job)
        self.pred_job = self.root.after(DELAY, self.predict)

    def redraw(self):
        self.canvas.delete("all")
        arr = np.array(self.img)

        for y in range(GRID):
            for x in range(GRID):
                v = arr[y, x]
                if v == 0:
                    continue

                if self.show_heatmap:
                    r = v
                    g = min(255, 255 - abs(v - 128))
                    b = 255 - v
                else:
                    r = g = b = v

                self.canvas.create_rectangle(
                    x*SCALE, y*SCALE,
                    (x+1)*SCALE, (y+1)*SCALE,
                    fill=f"#{r:02x}{g:02x}{b:02x}",
                    outline=""
                )

    def preprocess(self):
        img = self.img.filter(ImageFilter.GaussianBlur(0.8))
        arr = np.array(img, dtype=np.float32)
        arr[arr < 20] = 0

        ys, xs = np.nonzero(arr)
        if len(xs) == 0:
            return np.zeros((784, 1))

        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        digit = arr[y0:y1+1, x0:x1+1]

        h, w = digit.shape
        scale = 20 / max(h, w)
        nh, nw = int(h * scale), int(w * scale)

        digit = Image.fromarray(digit).resize((nw, nh), Image.BILINEAR)
        digit = np.array(digit)

        canvas = np.zeros((28, 28))
        yoff = (28 - nh) // 2
        xoff = (28 - nw) // 2
        canvas[yoff:yoff+nh, xoff:xoff+nw] = digit

        cy, cx = np.argwhere(canvas > 0).mean(axis=0)
        canvas = np.roll(canvas, int(14 - cy), axis=0)
        canvas = np.roll(canvas, int(14 - cx), axis=1)

        return (canvas / 255).reshape(784, 1)

    def animBar(self, bar, y, start, end, color):
        steps = 6
        for i in range(steps):
            t = (i + 1) / steps
            t = 1 - (1 - t) ** 2
            w = start + (end - start) * t
            
            self.bar_canvas.after(
                i * 15,
                lambda w=w, c=color: (
                    self.bar_canvas.coords(bar, 35, y-7, 35 + w, y+7),
                    self.bar_canvas.itemconfig(bar, fill=c)
                )
            )

    def pulseBar(self, bar):
        for i in range(2):
            self.bar_canvas.after(i * 120, lambda: self.bar_canvas.itemconfig(bar, width=3))
            self.bar_canvas.after(i * 120 + 60, lambda: self.bar_canvas.itemconfig(bar, width=1))

    def predict(self):
        x = self.preprocess()
        _, _, _, a2 = forwProp(self.w1, self.b1, self.w2, self.b2, x)
        probs = a2.flatten()
        pred = np.argmax(probs)

        self.label.config(
            text=f"Prediction: {pred} ({probs[pred]*100:.2f}%)"
        )

        for i in range(10):
            target = int(220 * probs[i])
            y = 25 + i * 25
            x0, _, x1, _ = self.bar_canvas.coords(self.bars[i])
            current = x1 - x0
            color = ACCENT if i == pred else SOFT

            self.animBar(self.bars[i], y, current, target, color)
            self.bar_canvas.itemconfig(
                self.texts[i], text=f"{probs[i]*100:.1f}%"
            )

        self.pulseBar(self.bars[pred])

    def clear(self):
        self.canvas.delete("all")
        self.img = Image.new("L", (GRID, GRID), 0)
        self.label.config(text="Prediction: —")
        
        for i in range(10):
            y = 25 + i * 25
            
            self.bar_canvas.coords(self.bars[i], 35, y-7, 35, y+7)
            self.bar_canvas.itemconfig(self.texts[i], text="0.0%")
