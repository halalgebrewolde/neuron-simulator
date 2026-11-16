from __future__ import annotations

from collections.abc import Sequence


def build_network_diagram(
    steps: Sequence[tuple[str, str, str, str]],
    *,
    selected_node: str | None = None,
    selected_hidden_node: str | None = None,
    selected_channel: int | None = None,
    selected_hidden_neuron: int | None = None,
) -> str:
    """Build a styled, left-to-right diagram of the model's computation flow."""
    lines = [
        "digraph NeuralNetwork {",
        '  graph [rankdir=LR, bgcolor="#E1C17F", pad="0.18", nodesep="0.28", ranksep="0.42", splines=ortho];',
        '  node [shape=box, style="rounded,filled", color="#201A22", penwidth=2, fontname="Arial", fontsize=10, margin="0.16,0.11"];',
        '  edge [color="#99BFA7", penwidth=2, arrowsize=0.72];',
    ]

    for node_id, title, detail, kind in steps:
        if node_id == selected_node and selected_channel is not None:
            detail = f"{detail} · selected channel {selected_channel}"
        if node_id == selected_hidden_node and selected_hidden_neuron is not None:
            detail = f"{detail} · selected neuron {selected_hidden_neuron}"
        label = f"{title}\\n{detail}".replace('"', '\\"')
        if node_id == selected_node:
            fill = "#17449E"
            font = "#FFF5D8"
            border = "#17151B"
            width = 4
        elif node_id == selected_hidden_node:
            fill = "#A79AC3"
            font = "#211C2B"
            border = "#51426F"
            width = 3
        elif kind == "activation":
            fill = "#A9CBB3"
            font = "#20212A"
            border = "#29252C"
            width = 2
        elif kind == "hidden":
            fill = "#B9A9CD"
            font = "#211C2B"
            border = "#383043"
            width = 2
        elif kind in {"input", "output"}:
            fill = "#17449E"
            font = "#FFF5D8"
            border = "#17151B"
            width = 2
        elif kind == "operation":
            fill = "#EBD39E"
            font = "#27212A"
            border = "#514734"
            width = 2
        else:
            fill = "#D9C07F"
            font = "#25202A"
            border = "#28222B"
            width = 2

        lines.append(
            f'  {node_id} [label="{label}", fillcolor="{fill}", fontcolor="{font}", '
            f'color="{border}", penwidth={width}];'
        )

    for index in range(len(steps) - 1):
        lines.append(f"  {steps[index][0]} -> {steps[index + 1][0]};")
    lines.append("}")
    return "\n".join(lines)


MAX_VISIBLE_NEURONS = 64


def _dot_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _visible_units(
    units: Sequence[tuple[int, float]],
    total_count: int,
    *,
    selected_unit: int | None = None,
    rank_by_activation: bool = False,
) -> tuple[list[tuple[int, float]], int]:
    """Keep every unit up to the display limit, then summarize the overflow."""
    if total_count <= MAX_VISIBLE_NEURONS:
        return list(units[:total_count]), 0

    real_unit_limit = MAX_VISIBLE_NEURONS - 1
    available = list(units)
    if rank_by_activation and len(available) > real_unit_limit:
        visible = sorted(available, key=lambda item: abs(item[1]), reverse=True)[:real_unit_limit]
    else:
        visible = available[:real_unit_limit]

    if selected_unit is not None and visible and all(unit_id != selected_unit for unit_id, _ in visible):
        selected_value = next(
            (value for unit_id, value in available if unit_id == selected_unit),
            None,
        )
        if selected_value is not None:
            visible[-1] = (selected_unit, selected_value)

    visible.sort(key=lambda item: item[0])
    return visible, max(0, total_count - len(visible))


def build_neuron_schematic(
    input_values: Sequence[tuple[int, float]],
    hidden_layers: Sequence[tuple[str, Sequence[tuple[int, float]]]],
    output_values: Sequence[tuple[int, str, float]],
    *,
    input_count: int,
    output_count: int,
    selected_units: Sequence[tuple[str, int]] = (),
    selected_output_index: int | None = None,
) -> str:
    """Build a multi-layer diagram, showing all units until a layer is too large."""
    selected_unit_map = dict(selected_units)
    lines = [
        "digraph NeuronSchematic {",
        '  graph [rankdir=LR, bgcolor="#E1C17F", pad="0.14", nodesep="0.025", ranksep="0.58", splines=line];',
        '  node [shape=circle, fixedsize=true, width="0.23", height="0.23", style=filled, penwidth=1.5, fontname="Arial", fontsize=5, fontcolor="#28222A"];',
        '  edge [color="#63596A", penwidth=0.75, arrowsize=0.45];',
    ]
    columns: list[list[str]] = []

    def add_layer(
        *,
        cluster_id: str,
        title: str,
        units: Sequence[tuple[int, float]],
        total_count: int,
        color: str,
        font_color: str,
        selected_unit: int | None = None,
        output_labels: dict[int, str] | None = None,
        input_layer: bool = False,
    ) -> list[str]:
        visible, omitted_count = _visible_units(
            units,
            total_count,
            selected_unit=selected_unit,
            rank_by_activation=not input_layer,
        )
        shown_count = len(visible)
        summary = f"{shown_count:,} shown"
        if omitted_count:
            summary += f" + {omitted_count:,} summarized"
        lines.append(
            f'  subgraph cluster_{cluster_id} {{ label="{_dot_escape(title)} · {total_count:,} ({summary})"; '
            'color="#8B7957"; fontcolor="#28222A"; fontsize=10; rank=same;'
        )

        node_ids: list[str] = []
        for unit_id, value in visible:
            node_id = f"{cluster_id}_{unit_id}"
            node_ids.append(node_id)
            chosen = unit_id == selected_unit
            outline = "#E1C17F" if chosen else "#28222A"
            penwidth = 3 if chosen else 1.5
            node_label = str(unit_id) if cluster_id.startswith("hidden") else ""
            if output_labels is not None:
                output_label = output_labels.get(unit_id, f"class {unit_id}")
                tooltip = f"{output_label} · {value:.2%}"
            elif input_layer:
                tooltip = f"Input value {unit_id} · {value:.3f}"
            else:
                tooltip = f"Unit {unit_id} · activation {value:.3f}"
            lines.append(
                f'    {node_id} [label="{node_label}", fillcolor="{color}", color="{outline}", '
                f'fontcolor="{font_color}", penwidth={penwidth}, tooltip="{_dot_escape(tooltip)}"];'
            )

        if omitted_count:
            more_id = f"{cluster_id}_more"
            node_ids.append(more_id)
            lines.append(
                f'    {more_id} [label="+", fillcolor="#EBD39E", color="#63596A", '
                f'fontcolor="#28222A", tooltip="{omitted_count:,} additional units summarized"];'
            )
        lines.append("  }")
        return node_ids

    columns.append(
        add_layer(
            cluster_id="input",
            title="Input values",
            units=input_values,
            total_count=input_count,
            color="#A79AC3",
            font_color="#28222A",
            input_layer=True,
        )
    )

    for index, (layer_label, units) in enumerate(hidden_layers):
        columns.append(
            add_layer(
                cluster_id=f"hidden_{index}",
                title=layer_label,
                units=units,
                total_count=len(units),
                color="#17449E",
                font_color="#FFF5D8",
                selected_unit=selected_unit_map.get(layer_label),
            )
        )

    output_units = [(unit_id, probability) for unit_id, _label, probability in output_values]
    output_labels = {unit_id: label for unit_id, label, _probability in output_values}
    columns.append(
        add_layer(
            cluster_id="output",
            title="Output classes",
            units=output_units,
            total_count=output_count,
            color="#A9CBB3",
            font_color="#28222A",
            selected_unit=selected_output_index,
            output_labels=output_labels,
        )
    )

    for source_nodes, target_nodes in zip(columns, columns[1:]):
        if not source_nodes or not target_nodes:
            continue
        links_per_node = min(3, len(target_nodes))
        for source_index, source_node in enumerate(source_nodes):
            linked_targets = set()
            for offset in range(links_per_node):
                target_node = target_nodes[(source_index * 3 + offset) % len(target_nodes)]
                if target_node not in linked_targets:
                    linked_targets.add(target_node)
                    lines.append(f"  {source_node} -> {target_node};")

    lines.append("}")
    return "\n".join(lines)
