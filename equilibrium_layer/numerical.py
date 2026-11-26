"""Dense direct oracle and central finite-difference checks for small graphs."""

from __future__ import annotations

from typing import Any

import numpy as np

from .backward import compute_parameter_vjps
from .errors import UnsupportedGraphError, ValidationError
from .forward import ForwardReceipt, solve_equilibrium
from .gradients import ParameterGradients
from .operators import dense_jacobian
from .upstream import align_upstream


def direct_dense_implicit_gradients(
    forward: ForwardReceipt,
    upstream: Any,
) -> tuple[np.ndarray, ParameterGradients, np.ndarray, float]:
    """Analytic dense-Jacobian implicit solve for N<16 direct comparisons."""

    graph, params = forward.graph, forward.params
    if graph.num_nodes >= 16:
        raise UnsupportedGraphError("dense direct oracle requires fewer than 16 active nodes")
    if not forward.accepted or forward.official_state is None or forward.gate is None:
        raise ValidationError("forward must be accepted before direct implicit differentiation")

    g = align_upstream(graph, params, upstream)
    z = forward.official_state
    J = dense_jacobian(graph, params, forward.gate)
    nd = graph.num_nodes * params.state_dim
    system = (np.eye(nd) - J).T
    lam = np.linalg.solve(system, g.ravel())
    residual = float(np.linalg.norm(system @ lam - g.ravel()) / max(np.linalg.norm(g.ravel()), 1e-12))
    grad_x, grads = compute_parameter_vjps(graph, params, z, forward.gate, lam)
    return grad_x, grads, lam.reshape(z.shape), residual


def finite_difference_gradients(
    forward_or_graph: ForwardReceipt | Any,
    upstream: Any,
    *,
    eps: float = 1e-5,
    max_actions: int = 1000,
    check_x: bool = True,
) -> tuple[np.ndarray | None, ParameterGradients]:
    """Central differences of the equilibrium objective ``sum(G*Z*)``.

    Every perturbed parameter request reruns the core equilibrium solver; no
    autodiff or unrolled derivative history is used.  Intended only for small
    direct-comparison graphs.
    """

    if isinstance(forward_or_graph, ForwardReceipt):
        base = forward_or_graph
        graph = base.graph
        params = base.params
    else:
        raise ValidationError("pass the accepted base ForwardReceipt")

    g = align_upstream(graph, params, upstream)

    def objective(W: np.ndarray, U: np.ndarray, b: np.ndarray, X: np.ndarray) -> float:
        from .parameters import Parameters

        p = Parameters(W=W, U=U, b=b, generation=params.generation)
        old_x = graph.X
        # Graph is frozen; construct a light copy through object.__setattr__ is
        # ugly, so use from_arrays with compact active IDs and no validity.
        from .graph import Graph

        gx = Graph(
            node_ids=graph.node_ids,
            X=X,
            edge_u=graph.edge_u,
            edge_v=graph.edge_v,
            edge_weight=graph.edge_weight,
            graph_id=graph.graph_id,
            revision=graph.revision,
            sample_weight=graph.sample_weight,
            family_id=graph.family_id,
            original_node_count=graph.original_node_count,
            active_indices=graph.active_indices,
        )
        r = solve_equilibrium(gx, p, max_actions=max_actions, tighten=True)
        if not r.accepted or r.official_state is None:
            raise ValidationError(f"finite-difference perturbed solve failed: {r.reason}")
        return float(np.sum(g * r.official_state))

    W0, U0, b0, X0 = params.W.copy(), params.U.copy(), params.b.copy(), graph.X.copy()
    fdW = np.zeros_like(W0)
    fdU = np.zeros_like(U0)
    fdb = np.zeros_like(b0)

    for idx in np.ndindex(W0.shape):
        wp, wm = W0.copy(), W0.copy()
        wp[idx] += eps
        wm[idx] -= eps
        fdW[idx] = (objective(wp, U0, b0, X0) - objective(wm, U0, b0, X0)) / (2 * eps)

    for idx in np.ndindex(U0.shape):
        up, um = U0.copy(), U0.copy()
        up[idx] += eps
        um[idx] -= eps
        fdU[idx] = (objective(W0, up, b0, X0) - objective(W0, um, b0, X0)) / (2 * eps)

    for k in range(b0.shape[0]):
        bp, bm = b0.copy(), b0.copy()
        bp[k] += eps
        bm[k] -= eps
        fdb[k] = (objective(W0, U0, bp, X0) - objective(W0, U0, bm, X0)) / (2 * eps)

    fdX: np.ndarray | None = None
    if check_x:
        fdX = np.zeros_like(X0)
        for idx in np.ndindex(X0.shape):
            xp, xm = X0.copy(), X0.copy()
            xp[idx] += eps
            xm[idx] -= eps
            fdX[idx] = (objective(W0, U0, b0, xp) - objective(W0, U0, b0, xm)) / (2 * eps)

    return fdX, ParameterGradients(fdW, fdU, fdb)


def compare_gradients(
    analytical_shared: ParameterGradients,
    fd_shared: ParameterGradients,
    *,
    analytical_x: np.ndarray | None = None,
    fd_x: np.ndarray | None = None,
    rtol: float = 2e-4,
    atol: float = 2e-6,
) -> dict[str, Any]:
    """Componentwise and relative comparison used by numerical tests."""

    result: dict[str, Any] = {"ok": True, "fields": {}}
    for name in ("W", "U", "b"):
        a = getattr(analytical_shared, name)
        b = getattr(fd_shared, name)
        abs_err = float(np.max(np.abs(a - b)))
        rel = float(np.linalg.norm((a - b).ravel()) / max(np.linalg.norm(b.ravel()), 1e-12))
        close = bool(np.allclose(a, b, rtol=rtol, atol=atol))
        result["fields"][name] = {"max_abs_error": abs_err, "relative_error": rel, "close": close}
        result["ok"] &= close

    if analytical_x is not None and fd_x is not None:
        abs_err = float(np.max(np.abs(analytical_x - fd_x)))
        rel = float(
            np.linalg.norm((analytical_x - fd_x).ravel())
            / max(np.linalg.norm(fd_x.ravel()), 1e-12)
        )
        close = bool(np.allclose(analytical_x, fd_x, rtol=rtol, atol=atol))
        result["fields"]["X"] = {"max_abs_error": abs_err, "relative_error": rel, "close": close}
        result["ok"] &= close

    return result
