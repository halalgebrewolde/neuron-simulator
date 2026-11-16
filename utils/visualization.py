from __future__ import annotations

import math
from threading import RLock

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch

MATPLOTLIB_LOCK = RLock()
BLACKBOX_CMAP = LinearSegmentedColormap.from_list(
    "black_box_sand_blue",
    ["#172D68", "#17449E", "#A9CBB3", "#E1C17F", "#A79AC3"],
)
FIGURE_BACKGROUND = "#E1C17F"
PANEL_BACKGROUND = "#EBD39E"
TEXT_COLOR = "#28222A"


def _as_channel_stack(activation: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(activation, torch.Tensor):
        values = activation.detach().to("cpu").numpy()
    else:
        values = np.asarray(activation)

    if values.ndim == 4:
        values = values[0]
    if values.ndim != 3:
        raise ValueError("Expected activations with shape [channels, height, width].")
    return values


def plot_activation_grid(
    activation: torch.Tensor | np.ndarray,
    *,
    start_channel: int = 0,
    channels: int = 16,
    columns: int = 4,
) -> plt.Figure:
    """Render a page of activation channels, normalized independently per map."""
    values = _as_channel_stack(activation)
    if start_channel < 0 or start_channel >= values.shape[0]:
        raise ValueError("start_channel must refer to an existing activation channel.")

    end_channel = min(start_channel + channels, values.shape[0])
    displayed = values[start_channel:end_channel]
    rows = math.ceil(len(displayed) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * 2.55, rows * 2.45),
        squeeze=False,
        facecolor=FIGURE_BACKGROUND,
        layout="constrained",
    )

    for offset, axis in enumerate(axes.flat):
        axis.set_facecolor(PANEL_BACKGROUND)
        if offset >= len(displayed):
            axis.axis("off")
            continue
        feature_map = displayed[offset]
        low = float(np.nanmin(feature_map))
        high = float(np.nanmax(feature_map))
        if not np.isfinite(low) or not np.isfinite(high):
            axis.axis("off")
            continue
        if high <= low:
            high = low + 1e-8
        axis.imshow(feature_map, cmap=BLACKBOX_CMAP, interpolation="nearest", vmin=low, vmax=high)
        axis.set_title(f"CHANNEL {start_channel + offset:03d}", color=TEXT_COLOR, fontsize=8, pad=5)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)

    return figure


def plot_single_activation(activation: torch.Tensor | np.ndarray) -> plt.Figure:
    """Render one channel with a color scale tied to its raw activation values."""
    if isinstance(activation, torch.Tensor):
        values = activation.detach().to("cpu").numpy()
    else:
        values = np.asarray(activation)
    if values.ndim != 2:
        raise ValueError("Expected one activation channel with shape [height, width].")

    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("Activation channel contains no finite values to display.")
    if high <= low:
        high = low + 1e-8

    figure, axis = plt.subplots(figsize=(7.2, 5.2), facecolor=FIGURE_BACKGROUND, layout="constrained")
    axis.set_facecolor(PANEL_BACKGROUND)
    image = axis.imshow(values, cmap=BLACKBOX_CMAP, interpolation="nearest", vmin=low, vmax=high)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("#554c42")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    colorbar.outline.set_edgecolor("#554c42")
    return figure


def plot_neuron_activations(
    activations: torch.Tensor | np.ndarray,
    *,
    selected_neuron: int,
) -> plt.Figure:
    """Plot a dense layer's actual unit values without Streamlit's pandas charts."""
    if isinstance(activations, torch.Tensor):
        values = activations.detach().to("cpu").numpy().reshape(-1)
    else:
        values = np.asarray(activations).reshape(-1)
    if values.size == 0 or not 0 <= selected_neuron < values.size:
        raise ValueError("selected_neuron must refer to a value in the activation vector.")

    figure, axis = plt.subplots(figsize=(8.8, 3.0), facecolor=FIGURE_BACKGROUND, layout="constrained")
    axis.set_facecolor(PANEL_BACKGROUND)
    x_values = np.arange(values.size)
    axis.plot(x_values, values, color="#17449E", linewidth=1.3, alpha=0.9)
    axis.scatter(
        [selected_neuron],
        [values[selected_neuron]],
        color="#A79AC3",
        edgecolors="#28222A",
        linewidths=1.0,
        s=54,
        zorder=3,
    )
    axis.axhline(0, color="#75684F", linewidth=0.8, alpha=0.8)
    axis.set_xlabel("Neuron index", color=TEXT_COLOR)
    axis.set_ylabel("Value", color=TEXT_COLOR)
    axis.tick_params(colors=TEXT_COLOR, labelsize=8)
    axis.grid(axis="y", color="#857452", alpha=0.22, linewidth=0.7)
    for spine in axis.spines.values():
        spine.set_color("#75684F")
    return figure
