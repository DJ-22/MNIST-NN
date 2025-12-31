import tkinter as tk
from tkinter import simpledialog, messagebox, ttk
from model import initParam, forwProp, backProp, updParam, loadParam, saveParam
from train import pred, acc
from data import train_X, train_Y, test_X, test_Y, M
from test import testNN, plotAcc
from ui import DigitUI


BG = "#f5f7fa"
CARD = "#ffffff"
ACCENT = "#2563eb"
SOFT = "#93c5fd"
TEXT = "#1f2937"
SUBTEXT = "#6b7280"

class MenuUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("MNIST Neural Network")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)

        card = tk.Frame(self.root, bg=CARD, padx=30, pady=30)
        card.pack(padx=20, pady=20)

        tk.Label(
            card, text="MNIST Neural Network",
            font=("Segoe UI", 20, "bold"),
            fg=TEXT, bg=CARD
        ).pack(anchor="w")

        tk.Label(
            card, text="Choose an action",
            font=("Segoe UI", 12),
            fg=SUBTEXT, bg=CARD
        ).pack(anchor="w", pady=(0, 20))

        self.makeBtn(card, "Train Neural Network", self.trainUI)
        self.makeBtn(card, "Test Neural Network", self.test)
        self.makeBtn(card, "Draw a Digit", self.draw)

        self.root.mainloop()

    def makeBtn(self, parent, text, cmd):
        tk.Button(
            parent, text=text, command=cmd,
            bg=ACCENT, fg="white", relief="flat",
            font=("Segoe UI", 12),
            padx=18, pady=8, width=22
        ).pack(pady=8)

    def trainUI(self):
        steps = simpledialog.askinteger(
            "Training", "Enter number of training steps:",
            parent=self.root, minvalue=1
        )
        if steps is None:
            return

        lr = simpledialog.askfloat(
            "Training", "Enter learning rate:",
            parent=self.root, minvalue=1e-6
        )
        if lr is None:
            return

        self.trainWindow(steps, lr)

    def trainWindow(self, steps, lr):
        win = tk.Toplevel(self.root)
        win.title("Training")
        win.configure(bg=BG)
        win.resizable(False, False)

        card = tk.Frame(win, bg=CARD, padx=25, pady=25)
        card.pack(padx=20, pady=20)

        tk.Label(
            card, text="Training Neural Network",
            font=("Segoe UI", 16, "bold"),
            fg=TEXT, bg=CARD
        ).pack(anchor="w")

        self.step_lbl = tk.Label(
            card, text="Step: 0",
            font=("Segoe UI", 11),
            fg=SUBTEXT, bg=CARD
        )
        self.step_lbl.pack(anchor="w", pady=(10, 0))

        self.acc_lbl = tk.Label(
            card, text="Accuracy: 0.00%",
            font=("Segoe UI", 11, "bold"),
            fg=ACCENT, bg=CARD
        )
        self.acc_lbl.pack(anchor="w", pady=(4, 12))

        self.pbar = ttk.Progressbar(
            card, length=320, mode="determinate"
        )
        self.pbar.pack(pady=(0, 6))
        self.pbar["maximum"] = steps

        param = loadParam()
        if param:
            self.w1, self.b1, self.w2, self.b2 = param
        else:
            self.w1, self.b1, self.w2, self.b2 = initParam()

        self.cur_step = 0
        self.max_steps = steps
        self.lr = lr

        self.runStep()

    def runStep(self):
        if self.cur_step >= self.max_steps:
            saveParam(self.w1, self.b1, self.w2, self.b2)
            messagebox.showinfo("Training", "Training completed!")
            return

        z1, a1, z2, a2 = forwProp(
            self.w1, self.b1, self.w2, self.b2, train_X
        )
        dw1, db1, dw2, db2 = backProp(
            self.w1, z1, a1,
            self.w2, z2, a2,
            train_X, train_Y, M
        )

        self.w1, self.b1, self.w2, self.b2 = updParam(
            self.w1, dw1,
            self.b1, db1,
            self.w2, dw2,
            self.b2, db2,
            self.lr
        )

        if self.cur_step % 50 == 0:
            preds = pred(a2)
            accuracy = acc(preds, train_Y)

            self.acc_lbl.config(
                text=f"Accuracy: {accuracy * 100:.2f}%"
            )

        if self.cur_step % 50 == 0:
            saveParam(self.w1, self.b1, self.w2, self.b2)

        self.cur_step += 1
        self.step_lbl.config(text=f"Step: {self.cur_step}")
        self.pbar["value"] = self.cur_step

        self.root.after(1, self.runStep)

    def test(self):
        param = loadParam()
        if not param:
            messagebox.showerror(
                "Error",
                "No trained parameters found.\nTrain the model first."
            )
            return

        w1, b1, w2, b2 = param
        testNN(w1, b1, w2, b2, test_X, test_Y)
        plotAcc(w1, b1, w2, b2, test_X, test_Y)

    def draw(self):
        self.root.destroy()
        DigitUI()
