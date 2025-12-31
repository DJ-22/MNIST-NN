# MNIST Neural Network From Scratch

---

This project is a complete implementation of a **neural network for MNSIT digit recognition**, built entirely from scratch without using any deep learning frameworks.

It combines **Core Neural Network Maths**, **V****ectorized Numerical Computing**, and a **User Interface** that enables real-time digit drawing, prediction visualization, and dataset augmentation through user feedback.

The goal of this project is to demonstrate a strong understanding of **Neural Network internals**, **Backpropagation**, and **End-to-End Machine Learning System Design**, rather than relying on high-level libraries.

---

## Project Overview

* Fully implemented neural network using **NumPy**
* Manual forward and backward propagation
* Interactive training, testing, and inference workflow
* Human-in-the-loop dataset expansion
* Persistent model parameters with resume capability

---

## Model Architecture

Input Layer

* 784 neurons (28 × 28 pixel grayscale image)

Hidden Layer

* 64 neurons
* **ReLU** activation
* **He** initialization

Output Layer

* 10 neurons
* **Softmax** activation

Loss Function

* Categorical Cross-Entropy

Optimization

* Batch Gradient Descent

---

## Key Features

### Neural Network from Scratch

* No use of **TensorFlow**, **PyTorch**, **Keras**, or similar frameworks
* Explicit implementation of forward propagation, backpropagation, and parameter updates
* Fully vectorized operations for efficiency and numerical stability

### Training and Evaluation

* Configurable number of training steps and learning rate
* Automatic checkpointing of model parameters
* Resume training from saved parameters
* Overall accuracy reporting
* Per-digit accuracy analysis with visualization

### Interactive Digit Recognition Interface

* Mouse-based digit drawing on a 28 × 28 grid
* Real-time predictions with class confidence visualization
* Optional heatmap view of pixel intensity
* MNIST-style preprocessing pipeline to ensure compatibility with training data

### Human-in-the-Loop Learning

* Users can mark predictions as correct or incorrect
* Incorrect predictions allow manual label correction
* New samples are appended directly to the dataset
* Enables continuous dataset improvement and retraining

---

## Digit Preprocessing Pipeline

The drawing interface applies a preprocessing pipeline designed to closely match MNIST data characteristics:

1. Gaussian smoothing
2. Noise thresholding
3. Bounding box extraction
4. Aspect-Ratio preserving resizing
5. Centering within a 28 × 28 canvas
6. Center-of-Mass alignment
7. Normalization to the [0, 1] range

---

## Project Structure

<pre class="overflow-visible! px-0!" data-start="277" data-end="808"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(--spacing(9)+var(--header-height))] @w-xl/main:top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre!"><span><span>.
├── nn/
│   ├── main.py        </span><span># Application entry point</span><span>
│   ├── menu_ui.py     </span><span># Menu-Driven GUI</span><span>
│   ├── ui.py          </span><span># Digit drawing and Interface</span><span>
│   ├── model.py       </span><span># Neural Network implementation</span><span>
│   ├── train.py       </span><span># Training Logic</span><span>
│   ├── test.py        </span><span># Evaluation and Visualization</span><span>
│   └── data.py        </span><span># Data loading and preprocessing</span><span>
├── dataset/
│   ├── data.csv       </span><span># Training dataset (MNIST-Style)</span><span>
│   └── parameters.npz </span><span># Saved Model Parameters</span><span>
├── requirements.txt
└── README.md</span></span></code></div></div></pre>

---

## How to Run

Install dependencies:

<pre class="overflow-visible! px-0!" data-start="3710" data-end="3753"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(--spacing(9)+var(--header-height))] @w-xl/main:top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>pip install -r requirements.txt
</span></span></code></div></div></pre>

Run the application:

<pre class="overflow-visible! px-0!" data-start="3776" data-end="3802"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(--spacing(9)+var(--header-height))] @w-xl/main:top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>python main.py
</span></span></code></div></div></pre>

The menu allows you to:

* Train the neural network
* Test model performance
* Draw digits and observe predictions

---
