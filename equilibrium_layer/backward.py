"""Implicit equilibrium VJPs, budgeted adjoint solve and risk checks."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

from .compat import DictAccessMixin
from .errors import ValidationError
from .forward import ForwardReceipt
from .gradients import ParameterGradients
from .graph import BACKWARD_RTOL, Graph
from .operators import ActionCounter, JacobianTransposeOperator, dense_jacobian
from .parameters import Parameters
from .upstream import align_upstream


@dataclass(frozen=True)
class RiskReport(DictAccessMixin):
    method: str
    inverse_norm_bound: float
    risk_limit: float | None
    risk_ok: bool
    risk_metric: str = "inverse_norm"
    risk_value: float | None = None
    inverse_norm_limit_ok: bool = False
    amplification_excess_ok: bool = False
    condition_excess_ok: bool = False
    jacobian_norm_bound_2: float | None = None
    jacobian_inf_bound: float | None = None
    jacobian_1_bound: float | None = None
    contraction_row_euclidean: float | None = None
    inverse_norm_bound_row_euclidean: float | None = None
    condition_bound_or_value: float | None = None
    sigma_min_ims: float | None = None
    reasons: tuple[str, ...] = ()

    @property
    def inverse_norm_excess(self) -> float:
        return max(0.0, self.inverse_norm_bound - 1.0)

    @property
    def condition_excess(self) -> float | None:
        if self.condition_bound_or_value is None or not math.isfinite(self.condition_bound_or_value):
            return None
        return max(0.0, self.condition_bound_or_value - 1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "inverse_norm_bound": self.inverse_norm_bound,
            "inverse_norm_excess": self.inverse_norm_excess,
            "risk_limit": self.risk_limit,
            "risk_metric": self.risk_metric,
            "risk_value": self.risk_value,
            "risk_ok": self.risk_ok,
            "inverse_norm_limit_ok": self.inverse_norm_limit_ok,
            "amplification_excess_ok": self.amplification_excess_ok,
            "condition_excess_ok": self.condition_excess_ok,
            "jacobian_norm_bound_2": self.jacobian_norm_bound_2,
            "jacobian_inf_bound": self.jacobian_inf_bound,
            "jacobian_1_bound": self.jacobian_1_bound,
            "contraction_row_euclidean": self.contraction_row_euclidean,
            "inverse_norm_bound_row_euclidean": self.inverse_norm_bound_row_euclidean,
            "condition_bound_or_value": self.condition_bound_or_value,
            "sigma_min_ims": self.sigma_min_ims,
            "reasons": list(self.reasons),
        }


@dataclass
class LinearSolveReport(DictAccessMixin):
    converged: bool
    relative_residual: float
    residual_tolerance: float
    actions_used: int
    max_actions: int | None
    status: str
    reason: str | None = None
    lambda_state: np.ndarray | None = None


@dataclass
class GradientReceipt(DictAccessMixin):
    eligible: bool
    status: str
    graph: Graph
    params: Parameters
    forward_receipt_id: str | None
    upstream: np.ndarray
    linear: LinearSolveReport
    risk: RiskReport
    grad_X: np.ndarray | None = None
    gradients: ParameterGradients | None = None
    sample_id: str | None = None
    cohort_id: str | None = None
    reasons: tuple[str, ...] = ()
    upstream_fingerprint: str | None = None

    @property
    def trainable(self) -> bool:
        return self.eligible

    @property
    def can_enter_training(self) -> bool:
        return self.eligible

    @property
    def linear_residual_ok(self) -> bool:
        return self.linear.converged

    @property
    def inverse_risk_ok(self) -> bool:
        return self.risk.risk_ok

    @property
    def relative_residual(self) -> float:
        return self.linear.relative_residual

    @property
    def residual_tolerance(self) -> float:
        return self.linear.residual_tolerance

    @property
    def inverse_risk_bound(self) -> float:
        return self.risk.inverse_norm_bound

    @property
    def condition_risk(self) -> float | None:
        return self.risk.condition_bound_or_value

    @property
    def graph_id(self) -> str:
        return self.graph.graph_id

    @property
    def graph_revision(self) -> int:
        return self.graph.revision

    @property
    def graph_fingerprint(self) -> str:
        return self.graph.fingerprint

    @property
    def parameter_generation(self) -> int:
        return self.params.generation

    @property
    def parameter_fingerprint(self) -> str:
        return self.params.fingerprint

    @property
    def sample_weight(self) -> float:
        return self.graph.sample_weight

    @property
    def effective_weight(self) -> float:
        return self.graph.sample_weight if self.eligible and not self.graph.is_empty else 0.0

    @property
    def grad_X_active(self) -> np.ndarray | None:
        return self.grad_X

    @property
    def grad_X_full(self) -> np.ndarray | None:
        """Scatter X VJP back to the caller's original node-array order.

        Invalid rows are exact zeros.  ``grad_X`` itself uses the compact
        active-row convention shared with Z and the sparse operator.
        """
        if self.grad_X is None:
            return None
        graph = self.graph
        full = np.zeros((int(graph.original_node_count or graph.num_nodes), graph.input_dim),
                        dtype=np.float64)
        if graph.active_indices is not None:
            full[graph.active_indices] = self.grad_X
        else:
            full[: graph.num_nodes] = self.grad_X
        return full

    @property
    def grad_X_original(self) -> np.ndarray | None:
        return self.grad_X_full

    @property
    def grad_W(self) -> np.ndarray | None:
        return None if self.gradients is None else self.gradients.W

    @property
    def grad_U(self) -> np.ndarray | None:
        return None if self.gradients is None else self.gradients.U

    @property
    def grad_b(self) -> np.ndarray | None:
        return None if self.gradients is None else self.gradients.b

    @property
    def weighted_shared_gradients(self) -> ParameterGradients | None:
        return None if self.gradients is None else self.gradients * self.sample_weight

    def public_dict(self) -> dict[str, Any]:
        return {
            "eligible": self.eligible,
            "trainable": self.trainable,
            "can_enter_training": self.can_enter_training,
            "linear_residual_ok": self.linear_residual_ok,
            "inverse_risk_ok": self.inverse_risk_ok,
            "status": self.status,
            "reasons": list(self.reasons),
            "sample_id": self.sample_id or self.graph.graph_id,
            "graph_id": self.graph.graph_id,
            "revision": self.graph.revision,
            "graph_fingerprint": self.graph.fingerprint,
            "parameter_generation": self.parameter_generation,
            "sample_weight": self.sample_weight,
            "effective_weight": self.effective_weight,
            "cohort_id": self.cohort_id,
            "linear_residual": self.linear.relative_residual,
            "residual_tolerance": self.linear.residual_tolerance,
            "linear_actions_used": self.linear.actions_used,
            "linear_status": self.linear.status,
            "risk": self.risk.to_dict(),
            "has_gradients": self.gradients is not None,
        }


def jacobian_norm_bounds(
    graph: Graph,
    params: Parameters,
    gate: np.ndarray,
    *,
    inverse_risk_limit: float | None = None,
    dense_small: bool = True,
    risk_metric: str = "inverse_norm",
) -> RiskReport:
    """Independent inverse-risk certificate.

    Large sparse graphs get structured O(E+ND^2) norm bounds. Graphs with
    fewer than 16 active nodes additionally get an exact dense SVD value, as
    expressly allowed by the data contract for direct comparisons.
    """

    n, d = graph.num_nodes, params.state_dim
    reasons: list[str] = []

    row_abs_A = np.asarray(abs(graph.A).sum(axis=1), dtype=np.float64).ravel()
    abs_W = np.abs(params.W)
    col_abs_W = abs_W.sum(axis=0)  # sum_a |W[a,k]|
    j_inf = float(np.max(gate * row_abs_A[:, None] * col_abs_W[None, :])) if n else 0.0

    # H[i,a] = sum_k gate[i,k] |W[a,k]|; then |A|^T H gives the 1-norm columns.
    h = gate @ abs_W.T
    j1_matrix = abs(graph.AT) @ h
    j1 = float(np.max(j1_matrix)) if j1_matrix.size else 0.0
    j2_bound = float(math.sqrt(max(0.0, j1 * j_inf)))

    sigma_w = float(np.linalg.norm(params.W, ord=2))
    qmax = gate.max(axis=1) if n else np.zeros(0)
    c_row2 = float(np.max(qmax * row_abs_A * sigma_w)) if n else 0.0
    inv_row2 = float("inf") if c_row2 >= 1.0 else 1.0 / (1.0 - c_row2)

    inv2_bound = float("inf") if j2_bound >= 1.0 else 1.0 / (1.0 - j2_bound)
    cond_bound = float("inf") if j2_bound >= 1.0 else (1.0 + j2_bound) / (1.0 - j2_bound)

    inverse_value = inv2_bound
    condition_value = cond_bound
    sigma_min = None
    method = "structured_sparse_norm_bound"

    if dense_small and n < 16 and n > 0:
        try:
            J = dense_jacobian(graph, params, gate)
            singular = np.linalg.svd(np.eye(n * d) - J, compute_uv=False)
            sigma_min = float(singular[-1])
            if sigma_min > 0.0 and np.isfinite(sigma_min):
                inverse_value = 1.0 / sigma_min
                condition_value = float(singular[0] / sigma_min)
                method = "dense_svd_direct_certificate"
            else:
                inverse_value = float("inf")
                condition_value = float("inf")
                reasons.append("singular_implicit_system")
        except np.linalg.LinAlgError:
            reasons.append("dense_risk_svd_failed")

    norm_ok = False
    excess_ok = False
    condition_excess_ok = False
    risk_value: float | None = None
    if inverse_risk_limit is None or not math.isfinite(float(inverse_risk_limit)):
        reasons.append("no_finite_inverse_risk_limit_supplied")
        risk_ok = False
    else:
        limit = float(inverse_risk_limit)
        norm_ok = math.isfinite(inverse_value) and inverse_value <= limit
        excess = max(0.0, inverse_value - 1.0) if math.isfinite(inverse_value) else float("inf")
        excess_ok = excess <= limit
        if condition_value is None or not math.isfinite(condition_value):
            cond_excess = float("inf")
        else:
            cond_excess = max(0.0, condition_value - 1.0)
        condition_excess_ok = cond_excess <= limit

        if risk_metric == "inverse_norm":
            risk_value = inverse_value
            risk_ok = norm_ok
        elif risk_metric == "row_euclidean_inverse_norm":
            risk_value = inv_row2
            risk_ok = math.isfinite(inv_row2) and inv_row2 <= limit
        elif risk_metric in ("amplification_excess", "inverse_excess"):
            risk_value = excess
            risk_ok = excess_ok
        elif risk_metric == "condition_excess":
            risk_value = cond_excess
            risk_ok = condition_excess_ok
        else:
            raise ValidationError(
                f"unknown risk_metric={risk_metric!r}; expected inverse_norm, "
                "row_euclidean_inverse_norm, amplification_excess or condition_excess"
            )
        if not risk_ok:
            reasons.append("inverse_norm_risk_exceeds_limit")

    return RiskReport(
        method=method,
        inverse_norm_bound=float(inverse_value),
        risk_limit=None if inverse_risk_limit is None else float(inverse_risk_limit),
        risk_ok=bool(risk_ok),
        risk_metric=risk_metric,
        risk_value=risk_value,
        inverse_norm_limit_ok=bool(norm_ok),
        amplification_excess_ok=bool(excess_ok),
        condition_excess_ok=bool(condition_excess_ok),
        jacobian_norm_bound_2=j2_bound,
        jacobian_inf_bound=j_inf,
        jacobian_1_bound=j1,
        contraction_row_euclidean=c_row2,
        inverse_norm_bound_row_euclidean=inv_row2,
        condition_bound_or_value=condition_value,
        sigma_min_ims=sigma_min,
        reasons=tuple(reasons),
    )


def _solve_adjoint_bicgstab(
    op: JacobianTransposeOperator,
    rhs: np.ndarray,
    max_actions: int,
    tol: float,
) -> LinearSolveReport:
    b = np.asarray(rhs, dtype=np.float64).ravel()
    bnorm = float(np.linalg.norm(b))
    if bnorm == 0.0:
        return LinearSolveReport(True, 0.0, tol, 0, max_actions, "accepted", None, np.zeros_like(b))

    counter = ActionCounter(max_actions=None)
    op.counter = counter
    target_tol = min(float(tol), 1e-10)

    x = np.zeros_like(b)
    r = b.copy()  # x0 = 0 => b - I*x = b; no operator action needed.
    rhat = r.copy()
    rho_prev = alpha = omega = 1.0
    v = np.zeros_like(b)
    p = np.zeros_like(b)
    certified_x: np.ndarray | None = None
    certified_res = float("inf")
    last_rel = 1.0
    status = "rejected"
    reason = "action_budget_exhausted"
    first = True

    while counter.vjp_actions < int(max_actions):
        rho = float(np.dot(rhat, r))
        if not np.isfinite(rho) or rho == 0.0:
            status = "breakdown"
            reason = "bicgstab_rho_breakdown"
            break

        if first:
            p = r.copy()
            first = False
        else:
            if not (np.isfinite(rho_prev) and rho_prev != 0.0 and np.isfinite(omega) and omega != 0.0):
                status = "breakdown"
                reason = "bicgstab_beta_breakdown"
                break
            beta = (rho / rho_prev) * (alpha / omega)
            if not np.isfinite(beta):
                status = "breakdown"
                reason = "bicgstab_beta_nonfinite"
                break
            p = r + beta * (p - omega * v)

        if counter.vjp_actions >= int(max_actions):
            reason = "action_budget_exhausted"
            break

        v = op.identity_minus_transpose_matvec(p)
        if not np.all(np.isfinite(v)):
            status = "breakdown"
            reason = "nonfinite_jacobian_action"
            break

        rhatv = float(np.dot(rhat, v))
        if not np.isfinite(rhatv) or rhatv == 0.0:
            status = "breakdown"
            reason = "bicgstab_alpha_breakdown"
            break
        alpha = rho / rhatv
        s = r - alpha * v
        s_norm = float(np.linalg.norm(s))
        last_rel = s_norm / bnorm
        x_half = x + alpha * p

        # s is directly b - A(x + alpha p) for the current residual r=b-Ax.
        if last_rel <= tol:
            certified_x = x_half.copy()
            certified_res = s_norm
            if last_rel <= target_tol:
                return LinearSolveReport(
                    True, last_rel, tol, counter.vjp_actions, max_actions, "accepted", None, certified_x
                )

        if counter.vjp_actions >= int(max_actions):
            reason = "action_budget_exhausted"
            break

        t = op.identity_minus_transpose_matvec(s)
        if not np.all(np.isfinite(t)):
            status = "breakdown"
            reason = "nonfinite_jacobian_action"
            break
        tnorm2 = float(np.dot(t, t))
        if tnorm2 == 0.0 or not np.isfinite(tnorm2):
            if certified_x is not None:
                break
            status = "breakdown"
            reason = "bicgstab_t_breakdown"
            break

        ts = float(np.dot(t, s))
        omega = ts / tnorm2
        if not np.isfinite(omega) or omega == 0.0:
            if certified_x is not None:
                break
            status = "breakdown"
            reason = "bicgstab_omega_breakdown"
            break

        x = x + alpha * p + omega * s
        r = s - omega * t
        r_norm = float(np.linalg.norm(r))
        last_rel = r_norm / bnorm

        if last_rel <= tol:
            # Independent explicit residual certificate when an action remains.
            if counter.vjp_actions < int(max_actions):
                ax = op.identity_minus_transpose_matvec(x)
                true_norm = float(np.linalg.norm(b - ax))
                true_rel = true_norm / bnorm
                last_rel = true_rel
                if true_rel <= tol:
                    certified_x = x.copy()
                    certified_res = true_norm
                    if true_rel <= target_tol:
                        return LinearSolveReport(
                            True, true_rel, tol, counter.vjp_actions, max_actions,
                            "accepted", None, certified_x
                        )
                r = b - ax
            else:
                certified_x = x.copy()
                certified_res = r_norm

        rho_prev = rho

    if certified_x is not None:
        rel = certified_res / bnorm
        return LinearSolveReport(
            converged=True,
            relative_residual=rel,
            residual_tolerance=tol,
            actions_used=counter.vjp_actions,
            max_actions=max_actions,
            status="accepted",
            reason=None if rel <= target_tol else "public_tolerance_certified",
            lambda_state=certified_x,
        )

    if reason == "action_budget_exhausted" and last_rel > tol:
        reason = "action_budget_exhausted_or_linear_residual_failed"
    return LinearSolveReport(
        converged=False,
        relative_residual=float(last_rel),
        residual_tolerance=tol,
        actions_used=counter.vjp_actions,
        max_actions=max_actions,
        status=status,
        reason=reason,
        lambda_state=None,
    )


def compute_parameter_vjps(
    graph: Graph,
    params: Parameters,
    z: np.ndarray,
    gate: np.ndarray,
    lambda_state: np.ndarray,
) -> tuple[np.ndarray, ParameterGradients]:
    lam = lambda_state.reshape(graph.num_nodes, params.state_dim)
    t = lam * gate
    grad_b = t.sum(axis=0)
    grad_U = graph.X.T @ t
    grad_X = t @ params.U.T
    az = graph.A @ z
    grad_W = az.T @ t
    return np.asarray(grad_X), ParameterGradients(W=np.asarray(grad_W), U=np.asarray(grad_U), b=np.asarray(grad_b))


def implicit_gradients(
    forward: ForwardReceipt,
    upstream: Any,
    *,
    max_actions: int = 100,
    inverse_risk_limit: float = 100.0,
    residual_tol: float = BACKWARD_RTOL,
    specified_nodes: Any | None = None,
    node_ids: Any | None = None,
    node_id: Any | None = None,
    sample_id: str | None = None,
    cohort_id: str | None = None,
    risk_metric: str = "inverse_norm",
) -> GradientReceipt:
    """Compute and independently qualify one sample's implicit VJP."""

    if node_id is not None:
        if specified_nodes is None and node_ids is None:
            if not isinstance(upstream, dict) and np.asarray(upstream).ndim == 1:
                upstream = {node_id: upstream}
            else:
                specified_nodes = [node_id]
    if specified_nodes is None and node_ids is not None:
        specified_nodes = node_ids

    if not isinstance(forward, ForwardReceipt):
        raise ValidationError("forward must be a ForwardReceipt from solve_equilibrium")

    graph, params = forward.graph, forward.params
    g = align_upstream(graph, params, upstream, specified_nodes=specified_nodes)

    import hashlib

    upstream_fp = hashlib.sha256(g.tobytes() + str(g.shape).encode()).hexdigest()

    if not forward.accepted or forward.official_state is None or forward.gate is None:
        linear = LinearSolveReport(
            False, float("inf"), residual_tol, 0, max_actions,
            "rejected", "forward_not_accepted", None
        )
        risk = jacobian_norm_bounds(
            graph, params, np.zeros((graph.num_nodes, params.state_dim)),
            inverse_risk_limit=inverse_risk_limit,
            risk_metric=risk_metric,
        )
        return GradientReceipt(
            eligible=False,
            status="rejected_forward",
            graph=graph,
            params=params,
            forward_receipt_id=forward.receipt_id,
            upstream=g,
            linear=linear,
            risk=risk,
            sample_id=sample_id,
            cohort_id=cohort_id,
            reasons=("forward_not_accepted",),
            upstream_fingerprint=upstream_fp,
        )

    if graph.is_empty:
        linear = LinearSolveReport(True, 0.0, residual_tol, 0, max_actions, "accepted_empty", None,
                                   np.zeros((0, params.state_dim)))
        risk = jacobian_norm_bounds(
            graph, params, forward.gate,
            inverse_risk_limit=inverse_risk_limit, risk_metric=risk_metric,
        )
        gx, grads = compute_parameter_vjps(graph, params, forward.official_state, forward.gate,
                                           np.zeros((0, params.state_dim)))
        return GradientReceipt(
            eligible=False,
            status="empty_excluded",
            graph=graph,
            params=params,
            forward_receipt_id=forward.receipt_id,
            upstream=g,
            linear=linear,
            risk=risk,
            grad_X=gx,
            gradients=grads,
            sample_id=sample_id,
            cohort_id=cohort_id,
            reasons=("empty_active_graph_not_counted_for_training",),
            upstream_fingerprint=upstream_fp,
        )

    risk = jacobian_norm_bounds(
        graph, params, forward.gate,
        inverse_risk_limit=inverse_risk_limit, risk_metric=risk_metric,
    )
    op = JacobianTransposeOperator(graph, params, forward.gate)
    linear = _solve_adjoint_bicgstab(
        op, g.ravel(), max_actions=int(max_actions), tol=float(residual_tol)
    )

    reasons: list[str] = []
    if not linear.converged:
        reasons.append("linear_residual_certificate_unavailable")
        if linear.reason:
            reasons.append(linear.reason)
    if not risk.risk_ok:
        reasons.extend(risk.reasons)

    if linear.converged and linear.lambda_state is not None and risk.risk_ok:
        gx, grads = compute_parameter_vjps(
            graph, params, forward.official_state, forward.gate, linear.lambda_state
        )
        return GradientReceipt(
            eligible=True,
            status="eligible",
            graph=graph,
            params=params,
            forward_receipt_id=forward.receipt_id,
            upstream=g,
            linear=linear,
            risk=risk,
            grad_X=gx,
            gradients=grads,
            sample_id=sample_id or graph.graph_id,
            cohort_id=cohort_id,
            reasons=(),
            upstream_fingerprint=upstream_fp,
        )

    return GradientReceipt(
        eligible=False,
        status="rejected_gradient",
        graph=graph,
        params=params,
        forward_receipt_id=forward.receipt_id,
        upstream=g,
        linear=linear,
        risk=risk,
        grad_X=None,
        gradients=None,
        sample_id=sample_id or graph.graph_id,
        cohort_id=cohort_id,
        reasons=tuple(dict.fromkeys(reasons)),
        upstream_fingerprint=upstream_fp,
    )


GradientResult = GradientReceipt
BackwardResult = GradientReceipt
