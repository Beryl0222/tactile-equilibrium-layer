"""Sparse named-node contact graphs and identity-safe warm starts."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import sparse

from .errors import ValidationError

# Public CPU envelope from docs/data-contract.md.  The small checked-in
# reference tensors have D=3 and are handled through the explicit direct /
# reference path; capacity-scale inputs use D in [8, 32].
SUPPORT = {
    "min_active_nodes": 1,
    "empty_active_nodes_allowed": True,
    "max_nodes": 2048,
    "max_edges_per_node": 12,
    "min_capacity_state_dim": 8,
    "max_state_dim": 32,
    "reference_min_state_dim": 1,
}

FORWARD_ATOL = 1e-8
FORWARD_RTOL = 1e-6
BACKWARD_RTOL = 1e-6


def _as_2d_finite(array: Any, name: str) -> np.ndarray:
    a = np.asarray(array, dtype=np.float64)
    if a.ndim != 2:
        raise ValidationError(f"{name} must be a 2D array, got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValidationError(f"{name} contains non-finite values")
    return a


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True)
class WarmStart:
    """A state whose rows carry stable node identities.

    A bare positional ndarray is intentionally not accepted as a warm start by
    the session: array positions are not identities across graph revisions.
    """

    node_ids: tuple[Any, ...]
    state: np.ndarray
    source_graph_id: str | None = None
    source_revision: int | None = None
    source_fingerprint: str | None = None
    parameter_generation: int | None = None

    def __post_init__(self) -> None:
        z = np.asarray(self.state, dtype=np.float64).copy()
        if z.ndim != 2 or z.shape[0] != len(self.node_ids):
            raise ValidationError("WarmStart state/name length mismatch")
        if not np.all(np.isfinite(z)):
            raise ValidationError("warm start contains non-finite values")
        object.__setattr__(self, "state", z)

    @property
    def state_dim(self) -> int:
        return int(self.state.shape[1])


def align_warm_start(
    graph: "Graph",
    warm_start: WarmStart | Mapping[Any, np.ndarray] | None,
    state_dim: int,
) -> tuple[np.ndarray, tuple[Any, ...], tuple[Any, ...]]:
    """Map an identity-bearing old state into this graph's active row order.

    Returns ``(z0, mapped_ids, unseen_current_ids)``.  Current nodes absent
    from the old state start at zero.  Old identities absent from the current
    graph are ignored.  Positions are never copied by index.
    """

    n = graph.num_nodes
    z0 = np.zeros((n, state_dim), dtype=np.float64)
    if warm_start is None:
        return z0, (), tuple(graph.node_ids)

    if isinstance(warm_start, WarmStart):
        source_ids = tuple(warm_start.node_ids)
        source_z = warm_start.state
    elif isinstance(warm_start, Mapping):
        source_ids = tuple(warm_start.keys())
        rows = [np.asarray(warm_start[k], dtype=np.float64).reshape(1, -1) for k in source_ids]
        source_z = np.concatenate(rows, axis=0) if rows else np.zeros((0, state_dim))
    else:
        raise ValidationError(
            "warm_start must carry node ids (WarmStart or mapping); positional arrays are not identity-safe"
        )

    if source_z.shape[1] != state_dim:
        raise ValidationError(
            f"warm start state dim {source_z.shape[1]} != parameter state dim {state_dim}"
        )
    if not np.all(np.isfinite(source_z)):
        raise ValidationError("warm start contains non-finite values")

    position = {node_id: i for i, node_id in enumerate(graph.node_ids)}
    mapped: list[Any] = []
    for source_id, row in zip(source_ids, source_z):
        target = position.get(source_id)
        if target is not None:
            z0[target] = row
            mapped.append(source_id)
    unseen = tuple(node_id for node_id in graph.node_ids if node_id not in set(mapped))
    return z0, tuple(mapped), unseen


@dataclass(frozen=True)
class Graph:
    """An immutable compact sparse graph containing only active nodes.

    Edge convention follows ``docs/data-contract.md``: edge ``(u, v, a)``
    contributes ``a * Z[v] @ W`` to row ``u``, i.e. ``A[u, v] += a``.
    Repeated ``(u, v)`` relations remain independent in the COO input and are
    summed by the sparse matrix representation.
    """

    node_ids: tuple[Any, ...]
    X: np.ndarray
    edge_u: np.ndarray
    edge_v: np.ndarray
    edge_weight: np.ndarray
    graph_id: str = "graph"
    revision: int = 1
    sample_weight: float = 1.0
    family_id: str | None = None
    original_node_count: int | None = None
    active_indices: np.ndarray | None = None
    valid: np.ndarray | None = None
    _fingerprint: str | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        ids = tuple(self.node_ids)
        if len(ids) != len(set(ids)):
            raise ValidationError("graph node names must be unique")

        X = _as_2d_finite(self.X, "X").copy()
        if X.shape[0] != len(ids):
            raise ValidationError("X row count must match active node count")

        u = np.asarray(self.edge_u, dtype=np.int64).reshape(-1).copy()
        v = np.asarray(self.edge_v, dtype=np.int64).reshape(-1).copy()
        w = np.asarray(self.edge_weight, dtype=np.float64).reshape(-1).copy()
        if not (len(u) == len(v) == len(w)):
            raise ValidationError("edge_u, edge_v and edge_weight must have equal length")
        n = len(ids)
        if u.size and (np.any(u < 0) or np.any(u >= n) or np.any(v < 0) or np.any(v >= n)):
            raise ValidationError("compact edge index outside active graph")
        if not np.all(np.isfinite(w)):
            raise ValidationError("edge weights contain non-finite values")

        weight = float(self.sample_weight)
        if not np.isfinite(weight) or weight < 0.0:
            raise ValidationError("sample_weight must be a finite non-negative number")

        if self.active_indices is None:
            active_indices = np.arange(n, dtype=np.int64)
        else:
            active_indices = np.asarray(self.active_indices, dtype=np.int64).reshape(-1).copy()
        if active_indices.shape[0] != n:
            raise ValidationError("active_indices length must equal active node count")

        original_n = int(self.original_node_count or n)
        valid = None
        if self.valid is not None:
            valid = np.asarray(self.valid, dtype=bool).reshape(-1).copy()
            if valid.shape[0] != original_n:
                raise ValidationError("valid length must equal original node count")
            if int(valid.sum()) != n:
                raise ValidationError("active nodes do not match valid mask")

        object.__setattr__(self, "node_ids", ids)
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "edge_u", u)
        object.__setattr__(self, "edge_v", v)
        object.__setattr__(self, "edge_weight", w)
        object.__setattr__(self, "sample_weight", weight)
        object.__setattr__(self, "active_indices", active_indices)
        object.__setattr__(self, "original_node_count", original_n)
        object.__setattr__(self, "valid", valid)

    @classmethod
    def from_arrays(
        cls,
        node_ids: Sequence[Any],
        X: Any,
        edge_u: Any | None = None,
        edge_v: Any | None = None,
        edge_weight: Any | None = None,
        *,
        edges: Iterable[Sequence[float]] | None = None,
        valid: Any | None = None,
        graph_id: str = "graph",
        revision: int = 1,
        sample_weight: float = 1.0,
        family_id: str | None = None,
    ) -> "Graph":
        raw_X = _as_2d_finite(X, "X")
        raw_n = len(node_ids)
        if raw_X.shape[0] != raw_n:
            raise ValidationError("X must have one row per declared node before validity filtering")

        if valid is None:
            mask = np.ones(raw_n, dtype=bool)
        else:
            mask = np.asarray(valid, dtype=bool).reshape(-1)
            if mask.shape[0] != raw_n:
                raise ValidationError("valid length must match node_ids")

        active_idx = np.flatnonzero(mask)
        old_to_new = {int(old): new for new, old in enumerate(active_idx)}

        eu: list[int] = []
        ev: list[int] = []
        ew: list[float] = []

        if edges is not None:
            for edge in edges:
                if len(edge) != 3:
                    raise ValidationError("each edge must be (u, v, weight)")
                a, b, weight = int(edge[0]), int(edge[1]), float(edge[2])
                if not (0 <= a < raw_n and 0 <= b < raw_n):
                    raise ValidationError("edge endpoint out of range")
                if mask[a] and mask[b]:
                    eu.append(old_to_new[a])
                    ev.append(old_to_new[b])
                    ew.append(weight)

        if edge_u is not None or edge_v is not None:
            u0 = np.asarray([] if edge_u is None else edge_u, dtype=np.int64).reshape(-1)
            v0 = np.asarray([] if edge_v is None else edge_v, dtype=np.int64).reshape(-1)
            w0 = np.asarray([] if edge_weight is None else edge_weight, dtype=np.float64).reshape(-1)
            if not (len(u0) == len(v0) == len(w0)):
                raise ValidationError("COO edge arrays must have equal lengths")
            for a, b, weight in zip(u0, v0, w0):
                a, b = int(a), int(b)
                if not (0 <= a < raw_n and 0 <= b < raw_n):
                    raise ValidationError("edge endpoint out of range")
                if mask[a] and mask[b]:
                    eu.append(old_to_new[a])
                    ev.append(old_to_new[b])
                    ew.append(float(weight))

        active_ids = tuple(node_ids[int(i)] for i in active_idx)
        return cls(
            node_ids=active_ids,
            X=raw_X[active_idx],
            edge_u=np.asarray(eu, dtype=np.int64),
            edge_v=np.asarray(ev, dtype=np.int64),
            edge_weight=np.asarray(ew, dtype=np.float64),
            graph_id=graph_id,
            revision=int(revision),
            sample_weight=float(sample_weight),
            family_id=family_id,
            original_node_count=raw_n,
            active_indices=active_idx,
            valid=mask,
        )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "Graph":
        graph_id = str(data.get("id", data.get("graph_id", "graph")))
        revision = int(data.get("revision", 1))
        family_id = data.get("family_id")
        return cls.from_arrays(
            node_ids=data["nodes"],
            X=data["X"],
            edges=data.get("edges"),
            edge_u=data.get("edge_u"),
            edge_v=data.get("edge_v"),
            edge_weight=data.get("edge_weight"),
            valid=data.get("valid"),
            graph_id=graph_id,
            revision=revision,
            sample_weight=float(data.get("sample_weight", data.get("weight", 1.0))),
            family_id=family_id,
        )

    @classmethod
    def from_edge_index(
        cls,
        node_ids: Sequence[Any],
        X: Any,
        edge_index: Any,
        edge_weight: Any | None = None,
        **kwargs: Any,
    ) -> "Graph":
        """PyG-style ``edge_index[0]=u, edge_index[1]=v`` construction path."""

        index = np.asarray(edge_index, dtype=np.int64)
        if index.shape[0] != 2:
            raise ValidationError("edge_index must have shape (2, E)")
        u, v = index[0], index[1]
        if edge_weight is None:
            edge_weight = np.ones(index.shape[1], dtype=np.float64)
        return cls.from_arrays(
            node_ids, X, edge_u=u, edge_v=v, edge_weight=edge_weight, **kwargs
        )

    @classmethod
    def from_adjacency(cls, node_ids: Sequence[Any], X: Any, A: Any, **kwargs: Any) -> "Graph":
        """Build from a sparse A. A dense ndarray is accepted only if caller materialized it."""

        coo = sparse.coo_matrix(A)
        return cls.from_arrays(
            node_ids,
            X,
            edge_u=coo.row,
            edge_v=coo.col,
            edge_weight=coo.data,
            **kwargs,
        )

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def input_dim(self) -> int:
        return int(self.X.shape[1])

    @property
    def num_edges(self) -> int:
        return int(self.edge_u.size)

    @property
    def is_empty(self) -> bool:
        return self.num_nodes == 0

    @property
    def out_degrees(self) -> np.ndarray:
        if self.num_nodes == 0:
            return np.zeros(0, dtype=np.int64)
        return np.bincount(self.edge_u, minlength=self.num_nodes).astype(np.int64)

    @property
    def max_out_degree(self) -> int:
        return int(self.out_degrees.max(initial=0))

    @property
    def max_abs_row_sum(self) -> float:
        if self.num_nodes == 0 or self.num_edges == 0:
            return 0.0
        return float(np.max(np.abs(self.A).sum(axis=1)))

    @property
    def A(self) -> sparse.csr_matrix:
        """Sparse accumulated A[u,v] += edge weight. Never dense."""
        if self.num_nodes == 0:
            return sparse.csr_matrix((0, 0), dtype=np.float64)
        coo = sparse.coo_matrix(
            (self.edge_weight, (self.edge_u, self.edge_v)),
            shape=(self.num_nodes, self.num_nodes),
            dtype=np.float64,
        )
        return coo.tocsr()

    @property
    def AT(self) -> sparse.csr_matrix:
        return self.A.T.tocsr()

    def dense_A(self, *, allow_large: bool = False) -> np.ndarray:
        """Direct-comparison helper.  Only allowed for fewer than 16 nodes."""
        if self.num_nodes >= 16 and not allow_large:
            raise UnsupportedGraphError(
                "dense adjacency materialization is restricted to graphs with fewer than 16 active nodes"
            )
        return self.A.toarray()

    def support_report(self, state_dim: int | None = None) -> dict[str, Any]:
        n = self.num_nodes
        problems: list[str] = []
        warnings: list[str] = []
        state_dim_hard_ok: bool | None = None
        capacity_state_dim_ok: bool | None = None
        if n == 0:
            warnings.append("empty_active_graph")
        elif n > SUPPORT["max_nodes"]:
            problems.append("too_many_nodes")
        if self.original_node_count is not None and self.original_node_count > SUPPORT["max_nodes"]:
            problems.append("original_graph_too_large")
        if self.max_out_degree > SUPPORT["max_edges_per_node"]:
            problems.append("too_many_edges_per_node")
        if state_dim is not None:
            d = int(state_dim)
            state_dim_hard_ok = (
                SUPPORT["reference_min_state_dim"] <= d <= SUPPORT["max_state_dim"]
            )
            capacity_state_dim_ok = (
                SUPPORT["min_capacity_state_dim"] <= d <= SUPPORT["max_state_dim"]
            )
            if not state_dim_hard_ok:
                problems.append("state_dim_outside_hard_support")
            elif not capacity_state_dim_ok:
                warnings.append("reference_small_state_dim_below_capacity_envelope")
        return {
            "supported": not problems,
            "declared_capacity_support": (
                not problems and state_dim is not None and bool(capacity_state_dim_ok)
            ),
            "within_node_envelope": n <= SUPPORT["max_nodes"],
            "within_degree_envelope": self.max_out_degree <= SUPPORT["max_edges_per_node"],
            "state_dim_hard_envelope_ok": state_dim_hard_ok,
            "state_dim_capacity_envelope_ok": capacity_state_dim_ok,
            "problems": tuple(problems),
            "warnings": tuple(warnings),
            "active_nodes": n,
            "original_nodes": self.original_node_count,
            "edges": self.num_edges,
            "max_out_degree": self.max_out_degree,
        }

    def snapshot(self) -> "Graph":
        """Return an isolated compact graph copy for an immutable forward receipt."""
        return Graph(
            node_ids=self.node_ids,
            X=self.X,
            edge_u=self.edge_u,
            edge_v=self.edge_v,
            edge_weight=self.edge_weight,
            graph_id=self.graph_id,
            revision=self.revision,
            sample_weight=self.sample_weight,
            family_id=self.family_id,
            original_node_count=self.original_node_count,
            active_indices=self.active_indices,
            valid=self.valid,
            _fingerprint=self._fingerprint,
        )

    @property
    def storage_bytes(self) -> int:
        """Bytes in project-owned compact arrays (not caller's read-only COO)."""
        total = self.X.nbytes + self.edge_u.nbytes + self.edge_v.nbytes + self.edge_weight.nbytes
        if self.active_indices is not None:
            total += self.active_indices.nbytes
        if self.valid is not None:
            total += self.valid.nbytes
        return int(total)

    @property
    def fingerprint(self) -> str:
        if self._fingerprint is not None:
            return self._fingerprint
        h = hashlib.sha256()
        h.update(b"qeq-graph-v1\n")
        h.update(
            _canonical_json(
                {
                    "id": self.graph_id,
                    "revision": self.revision,
                    "ids": [str(x) for x in self.node_ids],
                    "sample_weight": self.sample_weight,
                    "family_id": self.family_id,
                    "original_node_count": self.original_node_count,
                }
            )
        )
        for name, a in (
            ("X", self.X),
            ("u", self.edge_u),
            ("v", self.edge_v),
            ("w", self.edge_weight),
            ("active", self.active_indices),
        ):
            h.update(name.encode())
            h.update(str(a.shape).encode())
            h.update(str(a.dtype).encode())
            h.update(np.ascontiguousarray(a).tobytes())
        value = h.hexdigest()
        object.__setattr__(self, "_fingerprint", value)
        return value

    def summary(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "revision": self.revision,
            "fingerprint": self.fingerprint[:16],
            "active_nodes": self.num_nodes,
            "edges": self.num_edges,
            "sample_weight": self.sample_weight,
            "max_abs_row_sum": self.max_abs_row_sum,
        }
