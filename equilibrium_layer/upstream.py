"""Upstream margin alignment by stable node identity."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .errors import ValidationError
from .graph import Graph
from .parameters import Parameters


def align_upstream(
    graph: Graph,
    params: Parameters,
    upstream: Any,
    *,
    specified_nodes: Sequence[Any] | None = None,
) -> np.ndarray:
    """Return an active-row ``(N, D)`` upstream margin.

    Accepts:
      * ``None`` or zeros;
      * an active-order ndarray;
      * a full original-order ndarray, which is validity-compressed;
      * a mapping ``node_id -> D-vector`` for losses on specified nodes;
      * ``(nodes, values)`` plus ``specified_nodes``.
    """

    n, d = graph.num_nodes, params.state_dim
    if upstream is None:
        return np.zeros((n, d), dtype=np.float64)

    if isinstance(upstream, tuple) and len(upstream) == 2 and specified_nodes is None:
        specified_nodes, values = upstream
        return align_upstream(graph, params, values, specified_nodes=specified_nodes)

    if specified_nodes is not None:
        nodes = tuple(specified_nodes)
        values = np.asarray(upstream, dtype=np.float64)
        out = np.zeros((n, d), dtype=np.float64)
        index = {node_id: i for i, node_id in enumerate(graph.node_ids)}
        if values.shape != (len(nodes), d):
            raise ValidationError(
                f"specified upstream shape {values.shape} != ({len(nodes)}, {d})"
            )
        for node_id, row in zip(nodes, values):
            if node_id not in index:
                raise ValidationError(f"upstream node {node_id!r} is not an active current node")
            out[index[node_id]] = row
        if not np.all(np.isfinite(out)):
            raise ValidationError("upstream contains non-finite values")
        return out

    if isinstance(upstream, Mapping):
        out = np.zeros((n, d), dtype=np.float64)
        index = {node_id: i for i, node_id in enumerate(graph.node_ids)}
        for node_id, row in upstream.items():
            if node_id not in index:
                raise ValidationError(f"upstream node {node_id!r} is not an active current node")
            arr = np.asarray(row, dtype=np.float64).reshape(-1)
            if arr.shape != (d,):
                raise ValidationError(f"upstream for {node_id!r} must have shape ({d},)")
            out[index[node_id]] = arr
        if not np.all(np.isfinite(out)):
            raise ValidationError("upstream contains non-finite values")
        return out

    arr = np.asarray(upstream, dtype=np.float64)
    if arr.shape == (n, d):
        out = arr
    elif graph.original_node_count is not None and arr.shape == (graph.original_node_count, d):
        if graph.active_indices is None:
            raise ValidationError("full-order upstream supplied without active indices")
        out = arr[graph.active_indices]
    else:
        raise ValidationError(
            f"upstream shape {arr.shape} must be ({n}, {d}) active or "
            f"({graph.original_node_count}, {d}) original order"
        )
    if not np.all(np.isfinite(out)):
        raise ValidationError("upstream contains non-finite values")
    return np.array(out, dtype=np.float64, copy=True)
