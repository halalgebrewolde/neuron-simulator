import torch
from torch import nn


class UserDigitCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.hidden = nn.Linear(32 * 7 * 7, 64)
        self.hidden_relu = nn.ReLU()
        self.output = nn.Linear(64, 10)

    @property
    def activation_layers(self) -> dict[str, nn.Module]:
        return {
            "My convolution 1 · pre-activation": self.conv1,
            "My convolution 1 · ReLU activation": self.relu1,
            "My convolution 2 · pre-activation": self.conv2,
            "My convolution 2 · ReLU activation": self.relu2,
        }

    @property
    def neuron_layers(self) -> dict[str, nn.Module]:
        return {
            "My hidden layer · pre-activation": self.hidden,
            "My hidden layer · ReLU activation": self.hidden_relu,
            "Output logits": self.output,
        }

    @property
    def architecture_steps(self) -> list[tuple[str, str, str, str]]:
        return [
            ("input", "Input image", "28 × 28 · grayscale", "input"),
            ("conv1", "My convolution 1", "16 channels · 28 × 28", "convolution"),
            ("relu1", "ReLU activation", "negative responses become 0", "activation"),
            ("pool1", "Max pool", "28 × 28 → 14 × 14", "operation"),
            ("conv2", "My convolution 2", "32 channels · 14 × 14", "convolution"),
            ("relu2", "ReLU activation", "negative responses become 0", "activation"),
            ("pool2", "Max pool", "14 × 14 → 7 × 7", "operation"),
            ("flatten", "Flatten", "32 × 7 × 7 = 1,568 values", "operation"),
            ("hidden", "My hidden layer", "64 weighted neurons", "hidden"),
            ("hidden_relu", "Hidden ReLU", "keeps positive responses", "activation"),
            ("output", "Output layer", "10 digit logits", "output"),
            ("softmax", "Softmax", "10 class probabilities", "output"),
        ]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = self.pool(self.relu1(self.conv1(image)))
        image = self.pool(self.relu2(self.conv2(image)))
        image = image.flatten(start_dim=1)
        image = self.hidden_relu(self.hidden(image))
        return self.output(image)
