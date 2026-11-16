# Neuron simulator

An interactive CNN Activation Explorer for learning how image classifiers process an image.

## Choose an AI model

- **ImageNet** — pretrained ResNet-18 for general object photos. Captures the initial convolution and four residual stages.
- **MNIST** — a small starter CNN trained in the app to classify handwritten digits from 0 to 9.
- **My code** — build a custom AI interface in the in-app Python editor. It can use any installed framework or task, and its code controls inputs, model loading or training, and outputs. Save it to `user_ai.py` to keep it across app sessions.

The MNIST starter trains locally using the MNIST training set and reports accuracy on the MNIST test set. Its first run downloads MNIST into `data/`; the dataset is cached, and the trained model is retained in the current app session. Digit uploads are centered, converted to grayscale, resized to 28 × 28, and normalized. Turn off background inversion if the digit is already white on a dark background.

The custom editor expects a `main(st)` function. The code runs in the local app process and can import packages installed in the active Python environment. It can define its own inputs, training, predictions, and visualizations; it is not constrained to the image classifier's MNIST/PyTorch contract. Install additional frameworks into the same environment before importing them.

The built-in neuron diagram separates convolution, dense, pooled, and output layers into columns. It shows all channels or neurons for layers up to 64 units; larger layers show representative circles and a summary node. Custom AI code can create its own visuals with Streamlit or any installed plotting library.

The app palette uses the reference's sand-gold background, royal-blue highlights, dark outlines, mint connectors, and lavender hidden-layer accents. Adjust the Streamlit theme in `.streamlit/config.toml` and the illustration-specific color tokens in the style block near the top of `app.py`.

Feature maps are numerical responses. A visualization can suggest what a channel responds to, but it cannot by itself establish a semantic label or explain a prediction.

## Run locally

Use Python 3.10 or newer. Create and activate a virtual environment, then install the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Start the Streamlit app:

```powershell
python -m streamlit run app.py
```

The first ImageNet analysis downloads ResNet-18 weights. The first MNIST training downloads the digit dataset. Both require internet access once; the app caches data for later runs.

## Project layout

```text
app.py                    Streamlit interface and model selection
model.py                  ResNet-18 inference and activation hooks
mnist_engine.py           Digit preprocessing, CNN training, and activation hooks
user_ai.py                Starter script for a custom AI interface
user_digit_model.py       Earlier PyTorch CNN starter template
utils/network_diagram.py  Model flow diagram builder
utils/user_ai_runtime.py  Custom AI source loader and entrypoint
utils/visualization.py    Feature-map plotting helpers
requirements.txt          Python dependencies
```

## Current scope

- Built-in models: JPG, PNG, and WebP image classification
- Custom code: any interface, task, and framework installed in the Python environment
- Automatic activations: built-in ImageNet and MNIST models; custom code supplies its own interpretation views
- Execution: user-authored Python runs in the local app process
- Runtime: CPU by default, or CUDA when PyTorch detects a compatible GPU

Grad-CAM and perturbation analysis are not automatic for custom code; add them from the custom AI editor when needed.
