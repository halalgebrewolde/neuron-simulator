from __future__ import annotations

from threading import RLock
from typing import Any

import torch
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18


class ActivationExplorer:
    """Run a pretrained ResNet-18 and capture selected intermediate outputs."""

    LAYERS = {
        "conv1": "Initial convolution · pre-activation",
        "stem_relu": "Stem ReLU activation",
        "layer1": "Residual layer 1 · activated",
        "layer2": "Residual layer 2 · activated",
        "layer3": "Residual layer 3 · activated",
        "layer4": "Residual layer 4 · activated",
    }

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=self.weights).to(self.device)
        self.model.eval()
        self.preprocess = self.weights.transforms()
        self._activations: dict[str, torch.Tensor] = {}
        self._neuron_activations: dict[str, torch.Tensor] = {}
        self._inference_lock = RLock()
        self._hooks = []

        modules = {
            "conv1": self.model.conv1,
            "stem_relu": self.model.relu,
            "layer1": self.model.layer1,
            "layer2": self.model.layer2,
            "layer3": self.model.layer3,
            "layer4": self.model.layer4,
        }
        for name, module in modules.items():
            self._hooks.append(module.register_forward_hook(self._capture_feature(name)))
        self._hooks.append(self.model.avgpool.register_forward_hook(self._capture_hidden))

        self.architecture_steps = [
            ("input", "Input image", "224 × 224 · RGB", "input"),
            ("conv1", "Stem convolution", "64 channels · pre-activation", "convolution"),
            ("stem_relu", "Stem ReLU", "clips negative responses", "activation"),
            ("maxpool", "Max pool", "112 × 112 → 56 × 56", "operation"),
            ("layer1", "Residual layer 1", "64 channels · skip connection", "residual"),
            ("layer2", "Residual layer 2", "128 channels · skip connection", "residual"),
            ("layer3", "Residual layer 3", "256 channels · skip connection", "residual"),
            ("layer4", "Residual layer 4", "512 channels · skip connection", "residual"),
            ("hidden", "Global average pool", "512 hidden features", "hidden"),
            ("classifier", "Linear classifier", "1,000 class logits", "output"),
            ("softmax", "Softmax", "1,000 class probabilities", "output"),
        ]

    def _capture_feature(self, name: str):
        def hook(_module: Any, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            if isinstance(output, torch.Tensor):
                # Copy to CPU immediately so cached results do not retain GPU memory.
                self._activations[name] = output[0].detach().to("cpu").clone()

        return hook

    def _capture_hidden(self, _module: Any, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        self._neuron_activations["Pooled hidden features"] = (
            output[0].detach().to("cpu").reshape(-1).clone()
        )

    def inspect(self, image: Image.Image) -> dict[str, Any]:
        """Return top predictions and CPU copies of the selected layer outputs."""
        input_tensor = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)

        # Hooks write into shared model state, so serialize calls on the cached model.
        with self._inference_lock:
            self._activations = {}
            self._neuron_activations = {}
            with torch.inference_mode():
                logits = self.model(input_tensor)
            activations = dict(self._activations)
            neuron_activations = dict(self._neuron_activations)

        probabilities = torch.softmax(logits[0], dim=0).detach().to("cpu")
        neuron_activations["Classifier logits"] = logits[0].detach().to("cpu").clone()
        neuron_activations["Softmax probabilities"] = probabilities
        top_probabilities, top_indices = torch.topk(probabilities, k=5)
        categories = self.weights.meta["categories"]
        predictions = [
            {
                "label": categories[index],
                "confidence": float(probability),
            }
            for probability, index in zip(top_probabilities.tolist(), top_indices.tolist())
        ]

        return {
            "predictions": predictions,
            "activations": activations,
            "neuron_activations": neuron_activations,
            "output_probabilities": probabilities,
        }
