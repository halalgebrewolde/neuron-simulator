from __future__ import annotations

import hashlib
import io
from pathlib import Path

from matplotlib import pyplot as plt
import streamlit as st
from PIL import Image, UnidentifiedImageError
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor

from mnist_engine import (
    DigitActivationExplorer,
    MNISTStarterCNN,
    prepare_digit_image,
    train_digit_model,
)
from model import ActivationExplorer
from utils.network_diagram import (
    MAX_VISIBLE_NEURONS,
    build_network_diagram,
    build_neuron_schematic,
)
from utils.user_ai_runtime import run_user_ai_source
from utils.visualization import (
    MATPLOTLIB_LOCK,
    plot_activation_grid,
    plot_neuron_activations,
    plot_single_activation,
)


st.set_page_config(
    page_title="Neuron simulator",
    page_icon=":material/neurology:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    :root { --ink: #28222a; --muted: #63596a; --line: #554c42; --panel: #ead39e; --accent: #17449e; --mint: #a9cbb3; --lavender: #a79ac3; --sand: #e1c17f; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: radial-gradient(ellipse at 82% 0%, #f0d9a2 0%, #e1c17f 52%, #d9b76f 100%); color: var(--ink); }
    [data-testid="stHeader"] { background: transparent; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -.035em; }
    .eyebrow { color: #635176; font: 500 11px 'DM Mono', monospace; letter-spacing: .15em; text-transform: uppercase; }
    .hero-title { color: var(--ink); font: 600 clamp(2.5rem, 5vw, 4.5rem)/1.02 'Space Grotesk', sans-serif; letter-spacing: -.065em; margin: .8rem 0 .7rem; }
    .hero-title span { color: var(--accent); }
    .hero-copy { max-width: 760px; color: #4f4555; font-size: 1.05rem; line-height: 1.65; margin-bottom: 1.9rem; }
    .section-label { font: 500 11px 'DM Mono', monospace; letter-spacing: .1em; color: var(--muted); text-transform: uppercase; }
    .panel-title { font: 600 1.05rem 'Space Grotesk', sans-serif; color: var(--ink); margin: 0 0 .2rem; }
    .panel-subtitle { font-size: .86rem; color: var(--muted); margin: 0; }
    .step-card { height: 100%; padding: 1rem 1.05rem; border: 1px solid #655842; border-radius: 12px; background: rgba(240, 217, 162, .7); }
    .step-number { color: var(--accent); font: 500 11px 'DM Mono', monospace; }
    .step-heading { color: var(--ink); font: 600 1rem 'Space Grotesk', sans-serif; margin: .55rem 0 .2rem; }
    .step-text { color: #5b5260; font-size: .84rem; line-height: 1.45; }
    .empty-state { border: 1px dashed #766345; border-radius: 16px; padding: 2rem; background: rgba(240, 217, 162, .56); }
    .empty-state h3 { margin-top: 0; }
    .mono { font-family: 'DM Mono', monospace; }
    [data-testid="stMetric"] { background: rgba(240, 217, 162, .74); border: 1px solid #685a43; padding: .8rem 1rem; border-radius: 11px; }
    [data-testid="stMetricLabel"] { color: #5c5260; }
    [data-testid="stMetricValue"] { color: var(--ink); font-family: 'Space Grotesk', sans-serif; }
    div[data-testid="stFileUploader"] section { border-color: #746344; background: rgba(240, 217, 162, .62); }
    div[data-testid="stFileUploader"] section:hover { border-color: var(--accent); }
    div[data-testid="stProgress"] > div > div { background-color: var(--accent); }
    .footer-note { color: #62596a; font-size: .78rem; border-top: 1px solid #8b7957; padding-top: 1rem; margin-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_resnet_explorer() -> ActivationExplorer:
    return ActivationExplorer()


@st.cache_resource(show_spinner=False)
def load_mnist_datasets(data_root: str):
    transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    train_dataset = MNIST(data_root, train=True, download=True, transform=transform)
    test_dataset = MNIST(data_root, train=False, download=True, transform=transform)
    return train_dataset, test_dataset


def load_digit_explorer(
    epochs: int,
    train_examples: int,
    learning_rate: float,
    data_root: str,
    _progress_callback=None,
) -> DigitActivationExplorer:
    train_dataset, test_dataset = load_mnist_datasets(data_root)
    digit_model = MNISTStarterCNN()
    trained_model, accuracy = train_digit_model(
        digit_model,
        train_dataset,
        test_dataset,
        epochs=epochs,
        train_examples=train_examples,
        learning_rate=learning_rate,
        progress_callback=_progress_callback,
    )
    return DigitActivationExplorer(trained_model, test_accuracy=accuracy)


RESNET_LAYER_DETAILS = {
    "Initial convolution · pre-activation": "The stem convolution produces weighted sums before batch normalization and ReLU.",
    "Stem ReLU activation": "The stem's rectified responses after batch normalization.",
    "Residual layer 1 · activated": "The first residual stage adds its learned transform to a skip path, then applies ReLU.",
    "Residual layer 2 · activated": "A deeper residual stage with more channels and smaller spatial maps.",
    "Residual layer 3 · activated": "A higher-level residual representation with 256 channels.",
    "Residual layer 4 · activated": "The final spatial residual representation before global average pooling.",
}
RESNET_LAYER_NAME_TO_KEY = {
    label: key for key, label in ActivationExplorer.LAYERS.items()
}
MODEL_OPTIONS = ["ImageNet", "MNIST", "My code"]
MODEL_DESCRIPTIONS = {
    "ImageNet": "ResNet-18 · classifies general photos into ImageNet categories.",
    "MNIST": "Starter CNN · learns to classify handwritten digits from 0 to 9.",
    "My code": "Build a custom AI with your own framework, task, and interface.",
}
MAPS_PER_GROUP = 16

st.session_state.setdefault("trained_digit_model_key", None)
st.session_state.setdefault("trained_digit_explorer", None)

with st.sidebar:
    st.markdown('<div class="eyebrow">Neuron simulator</div>', unsafe_allow_html=True)
    st.markdown("## Model explorer")
    st.caption("Interactive neural network study")
    st.divider()

    st.markdown('<div class="section-label">01 / Choose a model</div>', unsafe_allow_html=True)
    selected_model = st.segmented_control(
        "AI model",
        options=MODEL_OPTIONS,
        default="ImageNet",
        required=True,
        width="stretch",
    )
    st.caption(MODEL_DESCRIPTIONS[selected_model])

    training_requested = False
    training_settings: tuple[int, int, float] | None = None
    user_ai_path = Path(__file__).resolve().with_name("user_ai.py")

    if selected_model == "MNIST":
        st.divider()
        st.markdown('<div class="section-label">02 / Train on MNIST</div>', unsafe_allow_html=True)
        epochs = st.slider("Training epochs", min_value=1, max_value=6, value=2, step=1)
        train_examples = st.slider(
            "Training examples",
            min_value=5_000,
            max_value=60_000,
            value=10_000,
            step=5_000,
            format="%d",
        )
        learning_rate = st.select_slider(
            "Learning rate",
            options=[0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005],
            value=0.001,
            format_func=lambda rate: f"{rate:g}",
        )
        training_settings = (epochs, train_examples, learning_rate)
        st.caption("The first run downloads MNIST. Training runs locally in this Python environment.")
        training_requested = st.button(
            "Train this model",
            type="primary",
            width="stretch",
            key="train-digit-model",
        )

    if selected_model != "My code":
        st.divider()
        st.markdown('<div class="section-label">03 / Choose an image</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload an image to inspect",
            type=["jpg", "jpeg", "png", "webp"],
            help="For MNIST, upload one handwritten digit. Dark ink on a light background is inverted automatically.",
            label_visibility="collapsed",
        )
        invert_light_background = True
        if selected_model == "MNIST":
            invert_light_background = st.checkbox(
                "Invert dark ink on a light background",
                value=True,
                help="Turn this off for white digits on a dark background.",
            )


st.markdown('<div class="eyebrow">Interactive neural network explorer</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Neuron <span>simulator</span></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-copy">Explore a pretrained image classifier, train a digit CNN, or run your own AI code in this local app.</div>',
    unsafe_allow_html=True,
)

if selected_model == "My code":
    if "user_ai_source" not in st.session_state:
        try:
            st.session_state["user_ai_source"] = user_ai_path.read_text(encoding="utf-8")
        except OSError:
            st.session_state["user_ai_source"] = ""

    with st.container(border=True):
        st.markdown("### Build any AI")
        st.caption(
            "Write a `main(st)` function to build the interface, load or train your model, and process any input. "
            "Use any framework and task supported by packages installed in this app’s Python environment."
        )
        user_ai_source = st.text_area(
            "Python AI source",
            key="user_ai_source",
            height=460,
            help="Define main(st). Put Streamlit input widgets, model loading/training, and output inside that function.",
        )
        save_code_col, run_code_col, _ = st.columns([1, 1, 2])
        with save_code_col:
            save_code_requested = st.button("Save code", key="save-user-ai-code", width="stretch")
        with run_code_col:
            run_code_requested = st.button(
                "Run my AI",
                key="run-user-ai-code",
                type="primary",
                width="stretch",
            )
        if save_code_requested:
            try:
                user_ai_path.write_text(user_ai_source, encoding="utf-8")
                st.success("Saved to user_ai.py", icon=":material/check:")
            except OSError as error:
                st.error(f"Could not save the AI code: {error}")
        st.caption("Install additional packages into this app’s Python environment before importing them. Your code runs locally with the app’s permissions.")

    source_digest = hashlib.sha256(user_ai_source.encode("utf-8")).hexdigest()
    if run_code_requested:
        st.session_state["active_user_ai_digest"] = source_digest

    if st.session_state.get("active_user_ai_digest") != source_digest:
        st.info("Edit the starter or paste your AI code, then select **Run my AI**.")
    else:
        try:
            run_user_ai_source(user_ai_source, st)
        except Exception as error:
            st.error("Your AI code raised an error.")
            with st.expander("Technical details", expanded=True):
                st.exception(error)
    st.stop()


digit_explorer: DigitActivationExplorer | None = None
training_error: Exception | None = None
training_key = None
if selected_model == "MNIST" and training_settings is not None:
    epochs, train_examples, learning_rate = training_settings
    data_root = str(Path(__file__).resolve().parent / "data")
    training_key = (
        selected_model,
        epochs,
        train_examples,
        learning_rate,
    )

    if training_requested:
        progress_bar = st.progress(0.0, text="Preparing the MNIST data…", width="stretch")

        def report_training_progress(fraction: float, message: str) -> None:
            progress_bar.progress(
                min(max(fraction, 0.0), 1.0),
                text=message,
                width="stretch",
            )

        try:
            with st.spinner("Training the CNN on MNIST…"):
                digit_explorer = load_digit_explorer(
                    epochs,
                    train_examples,
                    learning_rate,
                    data_root,
                    _progress_callback=report_training_progress,
                )
            st.session_state["trained_digit_model_key"] = training_key
            st.session_state["trained_digit_explorer"] = digit_explorer
            progress_bar.progress(
                1.0,
                text=f"Ready · MNIST test accuracy {digit_explorer.test_accuracy:.1%}",
                width="stretch",
            )
        except Exception as error:
            training_error = error
            progress_bar.empty()
    elif st.session_state.get("trained_digit_model_key") == training_key:
        digit_explorer = st.session_state.get("trained_digit_explorer")

if training_error:
    st.error("The MNIST starter model could not be trained. Confirm the dataset is available and the PyTorch environment can load it.")
    with st.expander("Technical details"):
        st.code(f"{type(training_error).__name__}: {training_error}")


if uploaded_file is None:
    if selected_model == "ImageNet":
        st.markdown(
            """
            <div class="empty-state">
              <div class="eyebrow">Start an exploration</div>
              <h3>Give the network something to see.</h3>
              <p style="color:#a7b4ae; margin-bottom:0">Upload a photo from the left panel. ResNet-18 will classify it, then reveal feature maps from five internal stages.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif digit_explorer is None:
        st.info("Choose a digit model, click **Train this model** in the sidebar, then upload a handwritten digit.", icon=":material/school:")
    else:
        st.markdown(
            """
            <div class="empty-state">
              <div class="eyebrow">Your digit model is ready</div>
              <h3>Upload a handwritten digit.</h3>
              <p style="color:#a7b4ae; margin-bottom:0">The MNIST CNN expects one digit at a time. It will center the mark, resize it to 28 × 28 pixels, and show the convolution maps.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.write("")
    guide_cols = st.columns(3)
    guide_steps = [
        ("01", "Choose a model", "Compare ImageNet with a digit specialist, or train your own coded CNN."),
        ("02", "Train on MNIST", "The digit models learn from labeled examples of handwritten digits."),
        ("03", "Inspect channels", "See how convolution layers transform the input into feature maps."),
    ]
    for column, (number, heading, description) in zip(guide_cols, guide_steps):
        with column:
            st.markdown(
                f'<div class="step-card"><div class="step-number">{number}</div><div class="step-heading">{heading}</div><div class="step-text">{description}</div></div>',
                unsafe_allow_html=True,
            )
    st.markdown(
        '<div class="footer-note">Feature maps are numerical responses. They can suggest what a channel responds to, but do not prove a semantic meaning or explain a prediction on their own.</div>',
        unsafe_allow_html=True,
    )
    st.stop()


image_bytes = uploaded_file.getvalue()
image_id = hashlib.sha256(image_bytes).hexdigest()
try:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
except (UnidentifiedImageError, OSError, ValueError) as error:
    st.error("That file could not be opened as an image. Try a JPG, PNG, or WebP file.")
    with st.expander("Technical details"):
        st.code(f"{type(error).__name__}: {error}")
    st.stop()


digit_preview: Image.Image | None = None
if selected_model == "ImageNet":
    try:
        with st.spinner("Loading ResNet-18 and capturing its activations…"):
            explorer = load_resnet_explorer()
            result = explorer.inspect(image)
        model_title = "ResNet-18 · ImageNet"
        layer_name_to_key = RESNET_LAYER_NAME_TO_KEY
        layer_details = RESNET_LAYER_DETAILS
        architecture_steps = explorer.architecture_steps
        model_accuracy = None
    except Exception as error:
        st.error("ResNet-18 could not be loaded or run. The first launch needs to download the pretrained model weights.")
        with st.expander("Technical details"):
            st.code(f"{type(error).__name__}: {error}")
        st.stop()
else:
    if digit_explorer is None:
        st.info("Train or load this model from the sidebar before analyzing an image.", icon=":material/school:")
        st.stop()
    digit_preview, digit_tensor = prepare_digit_image(
        image,
        invert_light_background=invert_light_background,
    )
    result = digit_explorer.inspect(digit_tensor)
    model_title = "MNIST starter CNN" if selected_model == "MNIST" else "Your coded CNN"
    layer_name_to_key = {label: label for label in digit_explorer.layers}
    layer_details = {
        label: (
            "Raw weighted sums from a learned convolution filter, before ReLU."
            if "pre-activation" in label
            else "The convolution responses after ReLU; negative values are zeroed."
        )
        for label in digit_explorer.layers
    }
    architecture_steps = digit_explorer.architecture_steps
    model_accuracy = digit_explorer.test_accuracy


with st.sidebar:
    st.divider()
    st.markdown('<div class="section-label">04 / Inspect the network</div>', unsafe_allow_html=True)
    selected_layer = st.selectbox(
        "Layer",
        options=list(layer_name_to_key),
        index=min(1, len(layer_name_to_key) - 1),
        help="A convolution layer produces a stack of feature maps (channels).",
    )

layer_key = layer_name_to_key[selected_layer]
activation = result["activations"][layer_key]
channel_count, map_height, map_width = activation.shape
group_starts = list(range(0, channel_count, MAPS_PER_GROUP))
hidden_activations = result["neuron_activations"]
hidden_options = [
    label
    for label in hidden_activations
    if label not in {"Classifier logits", "Output logits", "Softmax probabilities"}
]

with st.sidebar:
    group_start = st.selectbox(
        "Feature map group",
        options=group_starts,
        format_func=lambda start: f"Channels {start}–{min(start + MAPS_PER_GROUP, channel_count) - 1}",
        key=f"map-group-{selected_model}-{layer_key}",
    )
    selected_channel = st.selectbox(
        "Inspect a channel",
        options=list(range(channel_count)),
        format_func=lambda channel: f"Channel {channel:03d}",
        key=f"inspect-channel-{selected_model}-{layer_key}",
    )
    selected_hidden_layer = st.selectbox(
        "Hidden representation",
        options=hidden_options,
        index=min(1, len(hidden_options) - 1),
        key=f"hidden-layer-{selected_model}",
    )
    hidden_values = hidden_activations[selected_hidden_layer]
    selected_hidden_neuron = st.selectbox(
        "Inspect a hidden neuron",
        options=list(range(hidden_values.numel())),
        format_func=lambda neuron: f"Neuron {neuron:03d}",
        key=f"hidden-neuron-{selected_model}-{selected_hidden_layer}",
    )

st.markdown(
    f'<div class="section-label">Input image &nbsp; / &nbsp; {image.width} × {image.height} px</div>',
    unsafe_allow_html=True,
)
metric_cols = st.columns(5 if model_accuracy is not None else 4)
best = result["predictions"][0]
with metric_cols[0]:
    st.metric("Top prediction", best["label"])
with metric_cols[1]:
    st.metric("Model confidence", f"{best['confidence']:.1%}")
with metric_cols[2]:
    st.metric("Selected layer", selected_layer)
with metric_cols[3]:
    st.metric("Activation shape", f"{channel_count} × {map_height} × {map_width}")
if model_accuracy is not None:
    with metric_cols[4]:
        st.metric("MNIST test accuracy", f"{model_accuracy:.1%}")

st.caption(f"Active model · {model_title}")

if selected_model == "ImageNet":
    selected_diagram_node = layer_key
elif "convolution 1" in selected_layer.lower():
    selected_diagram_node = "relu1" if "relu activation" in selected_layer.lower() else "conv1"
elif "convolution 2" in selected_layer.lower():
    selected_diagram_node = "relu2" if "relu activation" in selected_layer.lower() else "conv2"
else:
    selected_diagram_node = None

if selected_hidden_layer == "Pooled hidden features":
    selected_hidden_diagram_node = "hidden"
elif "relu activation" in selected_hidden_layer.lower():
    selected_hidden_diagram_node = "hidden_relu"
else:
    selected_hidden_diagram_node = "hidden"

with st.container(border=True):
    st.markdown("### Neural network diagram")
    st.caption("Data flows from left to right. Blue marks the selected feature stage, mint marks activation functions, and lavender marks hidden units.")
    st.graphviz_chart(
        build_network_diagram(
            architecture_steps,
            selected_node=selected_diagram_node,
            selected_hidden_node=selected_hidden_diagram_node,
            selected_channel=selected_channel,
            selected_hidden_neuron=selected_hidden_neuron,
        ),
        width="stretch",
        height=300,
        alt=f"Neural network architecture for {model_title}, from image input through hidden features to output probabilities",
    )

if selected_model == "ImageNet":
    sampled_input = explorer.preprocess(image).reshape(-1)
    output_labels = explorer.weights.meta["categories"]
    image_hidden_layers = [
        ("Stem channels", "stem_relu"),
        ("Residual stage 1 channels", "layer1"),
        ("Residual stage 2 channels", "layer2"),
        ("Residual stage 3 channels", "layer3"),
        ("Residual stage 4 channels", "layer4"),
    ]
    hidden_layers = [
        (
            layer_label,
            [
                (channel, float(value))
                for channel, value in enumerate(
                    result["activations"][layer_name].mean(dim=(1, 2)).tolist()
                )
            ],
        )
        for layer_label, layer_name in image_hidden_layers
    ]
    pooled_values = result["neuron_activations"]["Pooled hidden features"]
    hidden_layers.append(
        ("Pooled hidden neurons", [(index, float(value)) for index, value in enumerate(pooled_values.tolist())])
    )
    selected_layer_names = {
        "conv1": "Stem channels",
        "stem_relu": "Stem channels",
        "layer1": "Residual stage 1 channels",
        "layer2": "Residual stage 2 channels",
        "layer3": "Residual stage 3 channels",
        "layer4": "Residual stage 4 channels",
    }
    selected_diagram_units = [
        (selected_layer_names[layer_key], selected_channel),
        ("Pooled hidden neurons", selected_hidden_neuron),
    ]
else:
    sampled_input = digit_tensor.reshape(-1)
    output_labels = [f"Digit {digit}" for digit in range(result["output_probabilities"].numel())]
    convolution_1 = result["activations"]["Convolution 1 · ReLU activation"]
    convolution_2 = result["activations"]["Convolution 2 · ReLU activation"]
    hidden_layers = [
        (
            "Convolution 1 channels",
            [
                (channel, float(value))
                for channel, value in enumerate(convolution_1.mean(dim=(1, 2)).tolist())
            ],
        ),
        (
            "Convolution 2 channels",
            [
                (channel, float(value))
                for channel, value in enumerate(convolution_2.mean(dim=(1, 2)).tolist())
            ],
        ),
        (
            "Dense hidden neurons",
            [(index, float(value)) for index, value in enumerate(hidden_values.tolist())],
        ),
    ]
    selected_diagram_units = [("Dense hidden neurons", selected_hidden_neuron)]
    if "convolution 1" in selected_layer.lower():
        selected_diagram_units.append(("Convolution 1 channels", selected_channel))
    elif "convolution 2" in selected_layer.lower():
        selected_diagram_units.append(("Convolution 2 channels", selected_channel))

input_count = sampled_input.numel()
input_sample_limit = MAX_VISIBLE_NEURONS - 1 if input_count > MAX_VISIBLE_NEURONS else MAX_VISIBLE_NEURONS
input_sample_count = min(input_sample_limit, input_count)
if input_sample_count == 1:
    sample_positions = [0]
else:
    sample_positions = [
        round(index * (input_count - 1) / (input_sample_count - 1))
        for index in range(input_sample_count)
    ]
schematic_inputs = [
    (position, float(sampled_input[position]))
    for position in sample_positions
]
output_probabilities = result["output_probabilities"]
output_count = output_probabilities.numel()
schematic_outputs = [
    (index, output_labels[index], float(output_probabilities[index]))
    for index in range(min(output_count, len(output_labels)))
]
largest_layer = max(
    [input_count, output_count, *(len(units) for _label, units in hidden_layers)]
)
diagram_height = max(520, min(MAX_VISIBLE_NEURONS, largest_layer) * 18)
with st.container(border=True):
    st.markdown("### Neuron view")
    st.caption(
        f"Each column is a separate layer. All channels or neurons are shown for layers up to {MAX_VISIBLE_NEURONS}; "
        f"larger layers show up to {MAX_VISIBLE_NEURONS - 1} representative circles plus a summary node. "
        "Convolution columns represent channels; dense columns represent neurons. Connections are illustrative."
    )
    st.graphviz_chart(
        build_neuron_schematic(
            schematic_inputs,
            hidden_layers,
            schematic_outputs,
            input_count=input_count,
            output_count=output_count,
            selected_units=selected_diagram_units,
            selected_output_index=int(output_probabilities.argmax().item()),
        ),
        width="stretch",
        height=diagram_height,
        alt=f"Layered neuron diagram for {model_title}; every neuron is shown in layers up to {MAX_VISIBLE_NEURONS}, with larger layers summarized.",
    )

input_col, maps_col = st.columns([0.82, 1.65], gap="large")
with input_col:
    st.markdown('<div class="panel-title">The input</div>', unsafe_allow_html=True)
    st.caption(uploaded_file.name)
    st.image(image, width="stretch", alt="Uploaded image being classified by the selected neural network")
    if digit_preview is not None:
        st.caption("28 × 28 input after centering and background adjustment")
        st.image(
            digit_preview,
            width=128,
            alt="Preprocessed grayscale digit passed to the MNIST convolutional network",
        )
    st.markdown("**Top predictions**")
    for rank, prediction in enumerate(result["predictions"], start=1):
        st.markdown(
            f"<span class='mono' style='color:#63596a'>{rank:02d}</span> &nbsp; **{prediction['label']}** &nbsp; <span class='mono' style='color:#17449e'>{prediction['confidence']:.1%}</span>",
            unsafe_allow_html=True,
        )
        st.progress(prediction["confidence"], width="stretch")

with maps_col:
    st.markdown('<div class="panel-title">Inside the network</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="panel-subtitle">{selected_layer} &nbsp;·&nbsp; {layer_details[selected_layer]} &nbsp; <span class="mono">{channel_count} channels · {map_height} × {map_width}</span></div>',
        unsafe_allow_html=True,
    )
    st.write("")
    with MATPLOTLIB_LOCK:
        grid_fig = plot_activation_grid(
            activation,
            start_channel=group_start,
            channels=MAPS_PER_GROUP,
        )
        st.pyplot(
            grid_fig,
            width="stretch",
            alt=f"Activation maps for channels {group_start} through {min(group_start + MAPS_PER_GROUP, channel_count) - 1} in {selected_layer}",
        )
        plt.close(grid_fig)
    st.caption(
        f"Showing channels {group_start}–{min(group_start + MAPS_PER_GROUP, channel_count) - 1}. "
        "Each map is independently contrast-normalized so its pattern is visible; colors should not be compared as absolute strength between channels."
    )

st.divider()
detail_col, explanation_col = st.columns([1.25, 1], gap="large")
with detail_col:
    st.markdown(f"### Channel {selected_channel:03d}")
    st.caption(f"{selected_layer} · {map_height} × {map_width} spatial positions")
    with MATPLOTLIB_LOCK:
        channel_fig = plot_single_activation(activation[selected_channel])
        st.pyplot(
            channel_fig,
            width="stretch",
            alt=f"Raw activation values in channel {selected_channel} of {selected_layer}",
        )
        plt.close(channel_fig)

with explanation_col:
    selected_values = activation[selected_channel]
    st.markdown("### Reading this feature map")
    st.write(
        f"This channel contains **{map_height * map_width:,} activation values**. "
        "Brighter regions indicate larger responses for this channel at those spatial positions."
    )
    if "pre-activation" in selected_layer:
        st.latex(r"z_{k,i,j}=\sum_{c,u,v}w_{k,c,u,v}x_{c,i+u,j+v}+b_k")
        st.write("The filter slides across the image or previous feature maps. Each channel uses one learned filter and has one response value at each location.")
    elif "ReLU activation" in selected_layer:
        st.latex(r"a_{k,i,j}=\max(0,z_{k,i,j})")
        st.write("This activated channel keeps positive filter responses and turns negative responses into zero. Compare it with the matching pre-activation channel to see exactly what changed.")
    elif selected_model == "ImageNet" and selected_layer.startswith("Residual layer"):
        st.latex(r"y=\operatorname{ReLU}(F(x)+x)")
        st.write("A residual block adds its learned transformation to a shortcut of the input, then applies ReLU. The channels show the resulting spatial responses.")
    elif selected_layer == "Stem ReLU activation":
        st.latex(r"a=\max(0,z)")
        st.write("Batch-normalized stem responses pass through ReLU. Negative values become zero before the next stage.")
    else:
        st.write("This is a post-activation feature map produced by the selected network stage.")
    stat_cols = st.columns(4 if "ReLU activation" in selected_layer else 3)
    with stat_cols[0]:
        st.metric("Minimum", f"{selected_values.min().item():.3f}")
    with stat_cols[1]:
        st.metric("Maximum", f"{selected_values.max().item():.3f}")
    with stat_cols[2]:
        st.metric("Mean", f"{selected_values.mean().item():.3f}")
    if "ReLU activation" in selected_layer:
        with stat_cols[3]:
            active_fraction = float((selected_values > 0).float().mean())
            st.metric("Active positions", f"{active_fraction:.1%}")
    st.info(
        "A channel is a learned response, not a guaranteed object detector. Its pattern can suggest what it responds to, but a heatmap alone cannot prove what the feature means or why the model made its prediction.",
        icon=":material/info:",
    )

st.divider()
hidden_col, output_col = st.columns([1.25, 1], gap="large")
with hidden_col:
    st.markdown("### Hidden neurons")
    st.caption(f"{selected_hidden_layer} · {hidden_values.numel()} values")
    with MATPLOTLIB_LOCK:
        neuron_fig = plot_neuron_activations(
            hidden_values,
            selected_neuron=selected_hidden_neuron,
        )
        st.pyplot(
            neuron_fig,
            width="stretch",
            alt=f"Activation values across {hidden_values.numel()} units in {selected_hidden_layer}",
        )
        plt.close(neuron_fig)
    neuron_value = float(hidden_values[selected_hidden_neuron])
    neuron_value_label = "Neuron pre-activation value" if "pre-activation" in selected_hidden_layer else "Neuron activation value"
    st.metric(f"{neuron_value_label} · {selected_hidden_neuron}", f"{neuron_value:.4f}")
    if selected_model == "ImageNet":
        st.write("Global average pooling reduces each final feature channel to one hidden value. The 512 pooled values feed the final classifier.")
        st.latex(r"h_c=\frac{1}{H W}\sum_{i,j}a_{c,i,j}")
    elif "ReLU activation" in selected_hidden_layer:
        st.write("Each dense neuron combines all previous features with learned weights and a bias, then ReLU clips negative results.")
        st.latex(r"a_j=\max(0,\sum_i w_{ji}x_i+b_j)")
    else:
        st.write("Each hidden neuron computes a weighted sum of the previous layer's values plus a learned bias. This is the value before its activation function.")
        st.latex(r"z_j=\sum_i w_{ji}x_i+b_j")

with output_col:
    st.markdown("### Output layer")
    st.write("The output layer produces one score (logit) per class. Softmax turns those scores into probabilities that add up to 1.")
    st.latex(r"p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}")
    st.caption("The prediction list above shows the five highest probabilities. Confidence is the model's probability for that class, not a guarantee that it is correct.")

st.markdown(
    f'<div class="footer-note">{model_title} · activations captured with PyTorch forward hooks. '
    "The displayed prediction is the model’s highest-probability class.</div>",
    unsafe_allow_html=True,
)
