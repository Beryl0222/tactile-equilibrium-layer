"""Budgeted sparse operators.  No dense adjacency or state Jacobian here."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .graph import Graph
from .parameters import Parameters


@dataclass
class ActionCounter:
    max_actions: int | None = None
    f_evaluations: int = 0
    jvp_actions: int = 0
    vjp_actions: int = 0

    def charge_f(self) -> None:
        self.f_evaluations += 1
        self._check()

    def charge_jvp(self) -> None:
        self.jvp_actions += 1
        self._check()

    def charge_vjp(self) -> None:
        self.vjp_actions += 1
        self._check()

    def _check(self) -> None:
        if self.max_actions is not None and self.total_actions > self.max_actions:
            raise RuntimeError("operator action budget exceeded")

    @property
    def total_actions(self) -> int:
        return self.f_evaluations + self.jvp_actions + self.vjp_actions

    @property
    def remaining(self) -> int | None:
        if self.max_actions is None:
            return None
        return max(0, self.max_actions - self.total_actions)


def elementwise_inf_norm(z: np.ndarray) -> float:
    if z.size == 0:
        return 0.0
    return float(np.max(np.abs(z)))


def precompute_input_projection(graph: Graph, params: Parameters) -> np.ndarray:
    if graph.input_dim != params.input_dim:
        raise ValueError(
            f"graph input dim {graph.input_dim} != U input dim {params.input_dim}"
        )
    return graph.X @ params.U


def evaluate_fixed_point(
    graph: Graph,
    params: Parameters,
    z: np.ndarray,
    *,
    input_projection: np.ndarray | None = None,
    counter: ActionCounter | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``F(z)``, tanh gate ``1-F(z)^2`` and pre-activation.

    One call is one fixed-point operator action.
    """

    if counter is not None:
        counter.charge_f()
    if input_projection is None:
        input_projection = precompute_input_projection(graph, params)
    if z.shape != (graph.num_nodes, params.state_dim):
        raise ValueError("z has the wrong state shape")

    # A is sparse O(E) storage/matvec.  The multiplication order is
    # A @ (Z @ W), never a materialized N*D by N*D Jacobian.
    propagated = graph.A @ (z @ params.W)
    preactivation = propagated + input_projection + params.b
    fz = np.tanh(preactivation)
    gate = 1.0 - fz * fz
    return fz, gate, preactivation


class JacobianTransposeOperator:
    """Matrix-free ``J_F(z)^T`` and ``(I-J_F(z))^T`` actions."""

    def __init__(
        self,
        graph: Graph,
        params: Parameters,
        gate: np.ndarray,
        counter: ActionCounter | None = None,
    ) -> None:
        self.graph = graph
        self.params = params
        self.gate = gate
        self.counter = counter
        self.n = graph.num_nodes
        self.d = params.state_dim
        self.shape = (self.n * self.d, self.n * self.d)

    def jvp(self, v: np.ndarray, *, count: bool = True) -> np.ndarray:
        if count and self.counter is not None:
            self.counter.charge_jvp()
        V = np.asarray(v, dtype=np.float64).reshape(self.n, self.d)
        out = self.gate * (self.graph.A @ (V @ self.params.W))
        return out.ravel()

    def vjp(self, y: np.ndarray, *, count: bool = True) -> np.ndarray:
        if count and self.counter is not None:
            self.counter.charge_vjp()
        Y = np.asarray(y, dtype=np.float64).reshape(self.n, self.d)
        # See module derivation: T=gate*Y, feature adjoint T @ W.T, then A^T.
        feature_adjoint = (self.gate * Y) @ self.params.W.T
        out = self.graph.AT @ feature_adjoint
        return np.asarray(out).ravel()

    def identity_minus_transpose_matvec(self, x: np.ndarray) -> np.ndarray:
        # One application of (I-J)^T requires one J^T action.
        return np.asarray(x, dtype=np.float64).ravel() - self.vjp(x)

    def __matmul__(self, other: np.ndarray) -> np.ndarray:
        return self.identity_minus_transpose_matvec(other)


def dense_jacobian(graph: Graph, params: Parameters, gate: np.ndarray) -> np.ndarray:
    """Analytic dense Jacobian for direct tests on N<16 graphs."""
    if graph.num_nodes >= 16:
        from .errors import UnsupportedGraphError

        raise UnsupportedGraphError("dense state Jacobian is only for fewer than 16 nodes")
    A = graph.dense_A()
    n, d = graph.num_nodes, params.state_dim
    # J[(i,k),(j,a)] = gate[i,k] A[i,j] W[a,k]
    J = np.einsum("ik,ij,ak->ikja", gate, A, params.W, optimize=True)
    return J.reshape(n * d, n * d)
