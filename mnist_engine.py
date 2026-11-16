from __future__ import annotations

import sys
from threading import RLock
from types import ModuleType
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Subset


class MNISTStarterCNN(nn.Module):
    """A small two-convolution CNN for classifying MNIST digits."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.hidden_relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    @property
    def activation_layers(self) -> dict[str, nn.Module]:
        return {
            "Convolution 1 · pre-activation": self.conv1,
            "Convolution 1 · ReLU activation": self.relu1,
            "Convolution 2 · pre-activation": self.conv2,
            "Convolution 2 · ReLU activation": self.relu2,
        }

    @property
    def neuron_layers(self) -> dict[str, nn.Module]:
        return {
            "Hidden dense · pre-activation": self.fc1,
            "Hidden dense · ReLU activation": self.hidden_relu,
            "Output logits": self.fc2,
        }

    @property
    def architecture_steps(self) -> list[tuple[str, str, str, str]]:
        return [
            ("input", "Input image", "28 × 28 · grayscale", "input"),
            ("conv1", "Convolution 1", "16 channels · 28 × 28", "convolution"),
            ("relu1", "ReLU activation", "negative responses become 0", "activation"),
            ("pool1", "Max pool", "28 × 28 → 14 × 14", "operation"),
            ("conv2", "Convolution 2", "32 channels · 14 × 14", "convolution"),
            ("relu2", "ReLU activation", "negative responses become 0", "activation"),
            ("pool2", "Max pool", "14 × 14 → 7 × 7", "operation"),
            ("flatten", "Flatten", "32 × 7 × 7 = 1,568 values", "operation"),
            ("hidden", "Hidden dense layer", "64 weighted neurons", "hidden"),
            ("hidden_relu", "Hidden ReLU", "keeps positive responses", "activation"),
            ("output", "Output layer", "10 digit logits", "output"),
            ("softmax", "Softmax", "10 class probabilities", "output"),
        ]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = self.pool(self.relu1(self.conv1(image)))
        image = self.pool(self.relu2(self.conv2(image)))
        image = image.flatten(start_dim=1)
        image = self.hidden_relu(self.fc1(image))
        return self.fc2(image)


def load_user_digit_model(source_code: str, source_digest: str) -> nn.Module:
    """Load user-authored model code as a normal Python module."""
    module_name = f"user_digit_model_{source_digest[:16]}"
    module = ModuleType(module_name)
    module.__file__ = "user_digit_model.py"
    sys.modules[module_name] = module
    try:
        compiled_code = compile(source_code, module.__file__, "exec")
        exec(compiled_code, module.__dict__)
        model_class = getattr(module, "UserDigitCNN")
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    model = model_class()
    if not isinstance(model, nn.Module):
        raise TypeError("UserDigitCNN must be a torch.nn.Module.")
    return model


def prepare_digit_image(
    image: Image.Image,
    *,
    invert_light_background: bool = True,
) -> tuple[Image.Image, torch.Tensor]:
    """Convert an uploaded image into a centered, normalized 28×28 MNIST input."""
    grayscale = ImageOps.autocontrast(image.convert("L"))
    if invert_light_background:
        grayscale = ImageOps.invert(grayscale)

    # MNIST digits are white strokes on a dark square. Crop, square, and center a
    # user's uploaded mark so it is closer to that training format.
    thresholded = grayscale.point(lambda pixel: 255 if pixel > 24 else 0)
    bounds = thresholded.getbbox()
    if bounds:
        digit = grayscale.crop(bounds)
    else:
        digit = grayscale

    side = max(digit.width, digit.height, 1)
    square = Image.new("L", (side, side), color=0)
    square.paste(digit, ((side - digit.width) // 2, (side - digit.height) // 2))
    digit_20 = square.resize((20, 20), Image.Resampling.LANCZOS)
    mnist_image = Image.new("L", (28, 28), color=0)
    mnist_image.paste(digit_20, (4, 4))

    values = np.asarray(mnist_image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(values).unsqueeze(0).unsqueeze(0)
    tensor = (tensor - 0.1307) / 0.3081
    return mnist_image, tensor


class DigitActivationExplorer:
    """Capture convolution maps and digit predictions for a trained CNN."""

    def __init__(self, model: nn.Module, test_accuracy: float) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.test_accuracy = test_accuracy
        self.layers = getattr(self.model, "activation_layers", None)
        if not isinstance(self.layers, dict) or not self.layers:
            raise TypeError(
                "Your model must define an activation_layers property containing "
                "named convolution modules."
            )
        self.neuron_layers = getattr(self.model, "neuron_layers", {})
        self.architecture_steps = getattr(self.model, "architecture_steps", None)
        if not self.architecture_steps:
            raise TypeError("Your model must define architecture_steps for its network diagram.")

        self._activations: dict[str, torch.Tensor] = {}
        self._neuron_activations: dict[str, torch.Tensor] = {}
        self._inference_lock = RLock()
        self._hooks = []
        for label, module in self.layers.items():
            self._hooks.append(module.register_forward_hook(self._capture_feature(label)))
        for label, module in self.neuron_layers.items():
            self._hooks.append(module.register_forward_hook(self._capture_neurons(label)))

    def _capture_feature(self, label: str):
        def hook(_module: Any, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            if not isinstance(output, torch.Tensor) or output.ndim != 4:
                raise TypeError(
                    f"Activation layer '{label}' must output a [batch, channels, height, width] tensor."
                )
            self._activations[label] = output[0].detach().to("cpu").clone()

        return hook

    def _capture_neurons(self, label: str):
        def hook(_module: Any, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            if not isinstance(output, torch.Tensor) or output.ndim < 2:
                raise TypeError(f"Neuron layer '{label}' must output a batched tensor.")
            self._neuron_activations[label] = output[0].detach().to("cpu").reshape(-1).clone()

        return hook

    def inspect(self, image_tensor: torch.Tensor) -> dict[str, Any]:
        input_tensor = image_tensor.to(self.device)
        with self._inference_lock:
            self._activations = {}
            self._neuron_activations = {}
            with torch.inference_mode():
                logits = self.model(input_tensor)
            activations = dict(self._activations)
            neuron_activations = dict(self._neuron_activations)

        if logits.ndim != 2 or logits.shape[1] != 10:
            raise ValueError("The custom model must return 10 logits, one for each digit 0–9.")
        probabilities = torch.softmax(logits[0], dim=0).detach().to("cpu")
        top_probabilities, top_indices = torch.topk(probabilities, k=5)
        neuron_activations["Softmax probabilities"] = probabilities
        predictions = [
            {
                "label": f"Digit {digit}",
                "confidence": float(probability),
            }
            for probability, digit in zip(top_probabilities.tolist(), top_indices.tolist())
        ]
        return {
            "predictions": predictions,
            "activations": activations,
            "neuron_activations": neuron_activations,
            "output_probabilities": probabilities,
        }


def train_digit_model(
    model: nn.Module,
    train_dataset: Any,
    test_dataset: Any,
    *,
    epochs: int,
    train_examples: int,
    learning_rate: float,
    progress_callback: Callable[[float, str], None] | None = None,
) -> tuple[nn.Module, float]:
    """Train a digit model and return it with its MNIST test-set accuracy."""
    example_count = min(max(1, train_examples), len(train_dataset))
    subset = Subset(train_dataset, range(example_count))
    generator = torch.Generator().manual_seed(42)
    train_loader = DataLoader(
        subset,
        batch_size=128,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    callback_interval = max(1, len(train_loader) // 32)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_index, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach())

            if progress_callback and (batch_index % callback_interval == 0):
                fraction = (epoch + (batch_index + 1) / len(train_loader)) / epochs
                average_loss = running_loss / (batch_index + 1)
                progress_callback(
                    fraction,
                    f"Epoch {epoch + 1}/{epochs} · loss {average_loss:.3f}",
                )

    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for images, labels in test_loader:
            logits = model(images.to(device))
            correct += int((logits.argmax(dim=1).cpu() == labels).sum())
            total += int(labels.numel())

    accuracy = correct / max(total, 1)
    if progress_callback:
        progress_callback(1.0, f"Training complete · test accuracy {accuracy:.1%}")
    return model.to("cpu").eval(), accuracy
