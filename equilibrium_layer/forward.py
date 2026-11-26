"""Budgeted equilibrium fixed-point solving."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .compat import DictAccessMixin
from .errors import ValidationError
from .graph import FORWARD_ATOL, FORWARD_RTOL, Graph, WarmStart, align_warm_start
from .operators import ActionCounter, elementwise_inf_norm, evaluate_fixed_point, precompute_input_projection
from .parameters import Parameters


@dataclass
class ForwardReceipt(DictAccessMixin):
    accepted: bool
    status: str
    graph: Graph
    params: Parameters
    candidate_state: np.ndarray
    residual: float
    residual_bound: float
    actions_used: int
    max_actions: int | None
    reason: str | None = None
    official_state: np.ndarray | None = None
    gate: np.ndarray | None = None
    receipt_id: str | None = None
    mapped_warm_node_ids: tuple[Any, ...] = ()
    unseen_node_ids: tuple[Any, ...] = ()
    support: dict[str, Any] = field(default_factory=dict)
    iterations: int = 0

    @property
    def viewable(self) -> bool:
        """A qualified equilibrium may be shown as the reconstruction."""
        return self.accepted and self.official_state is not None

    @property
    def converged(self) -> bool:
        return self.accepted

    @property
    def residual_norm(self) -> float:
        return self.residual

    @property
    def public_residual_bound(self) -> float:
        return self.residual_bound

    @property
    def tolerance(self) -> float:
        return self.residual_bound

    @property
    def can_commit_warm_start(self) -> bool:
        return self.accepted and self.official_state is not None

    @property
    def state(self) -> np.ndarray | None:
        """Official equilibrium state; ``None`` for an uncertified candidate."""
        return self.official_state if self.accepted else None

    def _expand_active(self, array: np.ndarray | None) -> np.ndarray | None:
        if array is None:
            return None
        full = np.zeros(
            (int(self.graph.original_node_count or self.graph.num_nodes), self.params.state_dim),
            dtype=np.float64,
        )
        if self.graph.active_indices is not None:
            full[self.graph.active_indices] = array
        else:
            full[: self.graph.num_nodes] = array
        return full

    @property
    def state_full(self) -> np.ndarray | None:
        """Official state scattered to original array order; invalid rows zero."""
        return self._expand_active(self.state)

    @property
    def official_state_full(self) -> np.ndarray | None:
        return self._expand_active(self.official_state)

    @property
    def candidate_state_full(self) -> np.ndarray:
        full = self._expand_active(self.candidate_state)
        assert full is not None
        return full

    @property
    def z(self) -> np.ndarray | None:
        return self.state

    @property
    def z_candidate(self) -> np.ndarray:
        return self.candidate_state

    @property
    def candidate_z(self) -> np.ndarray:
        return self.candidate_state

    @property
    def graph_fingerprint(self) -> str:
        return self.graph.fingerprint

    @property
    def parameter_fingerprint(self) -> str:
        return self.params.fingerprint

    @property
    def parameter_generation(self) -> int:
        return self.params.generation

    @property
    def accepted_node_ids(self) -> tuple[Any, ...]:
        return tuple(self.graph.node_ids) if self.accepted else ()

    def node_state(self, node_id: Any) -> np.ndarray:
        if not self.accepted or self.official_state is None:
            raise ValidationError("forward is not accepted; no official node state")
        try:
            i = self.graph.node_ids.index(node_id)  # tuple index; graphs small at API boundary
        except ValueError as exc:
            raise KeyError(node_id) from exc
        return self.official_state[i].copy()

    @property
    def node_states(self) -> dict[Any, np.ndarray]:
        if not self.accepted or self.official_state is None:
            return {}
        return {node_id: self.official_state[i].copy() for i, node_id in enumerate(self.graph.node_ids)}

    @property
    def retained_bytes(self) -> int:
        total = int(self.candidate_state.nbytes)
        if self.official_state is not None and self.official_state is not self.candidate_state:
            total += int(self.official_state.nbytes)
        if self.gate is not None:
            total += int(self.gate.nbytes)
        return total

    def public_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "accepted": self.accepted,
            "viewable": self.viewable,
            "status": self.status,
            "reason": self.reason,
            "graph_id": self.graph.graph_id,
            "revision": self.graph.revision,
            "graph_fingerprint": self.graph.fingerprint,
            "parameter_generation": self.parameter_generation,
            "residual": self.residual,
            "residual_bound": self.residual_bound,
            "actions_used": self.actions_used,
            "max_actions": self.max_actions,
            "accepted_nodes": list(self.accepted_node_ids),
            "mapped_warm_nodes": list(self.mapped_warm_node_ids),
            "new_zero_start_nodes": list(self.unseen_node_ids),
            "retained_bytes": self.retained_bytes,
            "support": self.support,
        }


# Semantic alias used by some callers.
ForwardResult = ForwardReceipt


def _unsupported_result(
    graph: Graph,
    params: Parameters,
    max_actions: int | None,
    support: dict[str, Any],
    reason: str,
) -> ForwardReceipt:
    z = np.zeros((graph.num_nodes, params.state_dim), dtype=np.float64)
    return ForwardReceipt(
        accepted=False,
        status="unsupported",
        reason=reason,
        graph=graph,
        params=params,
        candidate_state=z,
        residual=float("inf"),
        residual_bound=FORWARD_ATOL,
        actions_used=0,
        max_actions=max_actions,
        support=support,
    )


def solve_equilibrium(
    graph: Graph | dict[str, Any],
    params: Parameters | dict[str, Any],
    *,
    max_actions: int = 100,
    atol: float = FORWARD_ATOL,
    rtol: float = FORWARD_RTOL,
    warm_start: WarmStart | dict[Any, np.ndarray] | None = None,
    initial_state: np.ndarray | None = None,
    z0: np.ndarray | None = None,
    tighten: bool = True,
    anderson_history: int = 4,
    receipt_id: str | None = None,
) -> ForwardReceipt:
    """Repeatedly correct Z until the public infinity residual certificate.

    The returned receipt never marks a budget-exhausted candidate as accepted.
    Warm starts are mapped by stable node identity, never by array position.
    """

    if isinstance(graph, dict):
        graph = Graph.from_mapping(graph)
    if isinstance(params, dict):
        params = Parameters.from_mapping(params)
    if not isinstance(graph, Graph) or not isinstance(params, Parameters):
        raise ValidationError("graph and params must be Graph/Parameters objects or mappings")
    # Receipts explain this exact input even if the caller later mutates the
    # objects that were passed in.  Constructors copy all numeric arrays.
    graph = graph.snapshot()
    params = params.snapshot()
    if graph.input_dim != params.input_dim:
        raise ValidationError("graph X input dimension does not match U")
    if max_actions is not None and int(max_actions) < 0:
        raise ValidationError("max_actions must be non-negative")

    support = graph.support_report(params.state_dim)
    hard_problems = [p for p in support["problems"] if p != "state_dim_outside_hard_support"]
    # State dim >32 or <1 is in problems; support report currently same code.
    if support["problems"]:
        return _unsupported_result(
            graph, params, max_actions, support, ",".join(support["problems"])
        )
    if hard_problems:
        return _unsupported_result(graph, params, max_actions, support, ",".join(hard_problems))

    if initial_state is None:
        initial_state = z0
    if initial_state is not None:
        aligned_z = np.asarray(initial_state, dtype=np.float64)
        if aligned_z.shape != (graph.num_nodes, params.state_dim):
            raise ValidationError(
                f"initial_state shape {aligned_z.shape} != "
                f"{(graph.num_nodes, params.state_dim)}"
            )
        if not np.all(np.isfinite(aligned_z)):
            raise ValidationError("initial_state contains non-finite values")
        aligned_z = aligned_z.copy()
        mapped = ()
        unseen = ()
    else:
        aligned_z, mapped, unseen = align_warm_start(graph, warm_start, params.state_dim)

    if graph.is_empty:
        z = np.zeros((0, params.state_dim), dtype=np.float64)
        return ForwardReceipt(
            accepted=True,
            status="accepted_empty",
            graph=graph,
            params=params,
            candidate_state=z,
            official_state=z,
            gate=np.ones_like(z),
            residual=0.0,
            residual_bound=0.0,
            actions_used=0,
            max_actions=max_actions,
            receipt_id=receipt_id,
            mapped_warm_node_ids=mapped,
            unseen_node_ids=unseen,
            support=support,
        )

    if max_actions == 0:
        # With no charged action the residual is intentionally unknown; do
        # not secretly evaluate F and violate the operator quota.
        residual = float("inf")
        return ForwardReceipt(
            accepted=False,
            status="rejected",
            reason="action_budget_exhausted",
            graph=graph,
            params=params,
            candidate_state=aligned_z,
            residual=residual,
            residual_bound=atol + rtol * elementwise_inf_norm(aligned_z),
            actions_used=0,
            max_actions=0,
            receipt_id=receipt_id,
            mapped_warm_node_ids=mapped,
            unseen_node_ids=unseen,
            support=support,
        )

    xp = precompute_input_projection(graph, params)
    counter = ActionCounter(max_actions=int(max_actions))
    z = aligned_z.copy()

    dx_hist: list[np.ndarray] = []
    dr_hist: list[np.ndarray] = []
    prev_x: np.ndarray | None = None
    prev_r: np.ndarray | None = None
    best: dict[str, Any] | None = None
    official: dict[str, Any] | None = None
    status = "rejected"
    reason = "action_budget_exhausted"
    iterations = 0

    tight_atol = 1e-12
    tight_rtol = 1e-10

    while counter.total_actions < int(max_actions):
        iterations += 1
        try:
            fz, gate, _ = evaluate_fixed_point(
                graph, params, z, input_projection=xp, counter=counter
            )
        except RuntimeError:
            reason = "action_budget_exhausted"
            break

        if not np.all(np.isfinite(fz)) or not np.all(np.isfinite(gate)):
            status = "rejected"
            reason = "nonfinite_operator"
            best = {"z": z, "fz": np.zeros_like(z), "gate": np.zeros_like(z),
                    "res": float("inf"), "bound": float("inf")}
            break

        r = fz - z
        residual = elementwise_inf_norm(z - fz)
        z_inf = elementwise_inf_norm(z)
        bound = float(atol + rtol * z_inf)
        tight_bound = tight_atol + tight_rtol * z_inf

        old_best_res = float("inf") if best is None else float(best["res"])
        if best is None or residual < old_best_res:
            best = {
                "z": z.copy(),
                "fz": fz.copy(),
                "gate": gate.copy(),
                "res": residual,
                "bound": bound,
            }

        if residual <= bound and official is None:
            official = {
                "z": z.copy(),
                "gate": gate.copy(),
                "res": residual,
                "bound": bound,
                "actions": counter.total_actions,
            }

        if official is not None and (not tighten or residual <= tight_bound):
            status = "accepted"
            reason = None
            official = {
                "z": z.copy(),
                "gate": gate.copy(),
                "res": residual,
                "bound": bound,
                "actions": counter.total_actions,
            }
            break

        # Anderson history transition from the preceding evaluated point.
        if prev_x is not None and prev_r is not None:
            dx = (z - prev_x).ravel()
            dr = (r - prev_r).ravel()
            if np.linalg.norm(dx) > 1e-14 and np.linalg.norm(dr) > 1e-14:
                dx_hist.append(dx)
                dr_hist.append(dr)
                if anderson_history > 0:
                    keep = int(anderson_history)
                    dx_hist = dx_hist[-keep:]
                    dr_hist = dr_hist[-keep:]

        # Safeguard a divergent Anderson/fixed iterate by returning to the
        # plain fixed-point image of the best evaluated point.
        if best is not None and residual > max(10.0 * old_best_res, 1e-6) and np.isfinite(old_best_res):
            z = best["fz"].copy()
            dx_hist.clear()
            dr_hist.clear()
            prev_x = None
            prev_r = None
            continue

        prev_x = z.copy()
        prev_r = r.copy()
        z_next = fz.copy()

        if anderson_history > 0 and dr_hist:
            k = min(int(anderson_history), len(dr_hist))
            dx_cols = dx_hist[-k:]
            dr_cols = dr_hist[-k:]
            try:
                # Small k x k normal system.  This avoids materializing an
                # N*D x k least-squares matrix or a state Jacobian.
                gram = np.empty((k, k), dtype=np.float64)
                rhs = np.empty(k, dtype=np.float64)
                for i, dri in enumerate(dr_cols):
                    rhs[i] = float(np.dot(dri, r.ravel()))
                    for j, drj in enumerate(dr_cols):
                        gram[i, j] = float(np.dot(dri, drj))
                diag_eps = 1e-12 * max(float(np.trace(gram)), 1.0)
                gamma = np.linalg.solve(gram + diag_eps * np.eye(k), rhs)
                correction = np.zeros_like(z.ravel())
                for coeff, dx_col, dr_col in zip(gamma, dx_cols, dr_cols):
                    correction += float(coeff) * (dx_col + dr_col)
                candidate = z.ravel() + r.ravel() - correction
                candidate = candidate.reshape(z.shape)
                if np.all(np.isfinite(candidate)) and elementwise_inf_norm(candidate) <= 10.0:
                    z_next = candidate
            except np.linalg.LinAlgError:
                z_next = fz.copy()

        z = z_next

    actions_used = counter.total_actions

    if status == "accepted" and official is not None:
        z_out = official["z"]
        return ForwardReceipt(
            accepted=True,
            status="accepted",
            graph=graph,
            params=params,
            candidate_state=z_out,
            official_state=z_out,
            gate=official["gate"],
            residual=float(official["res"]),
            residual_bound=float(official["bound"]),
            actions_used=actions_used,
            max_actions=max_actions,
            reason=None,
            receipt_id=receipt_id,
            mapped_warm_node_ids=mapped,
            unseen_node_ids=unseen,
            support=support,
            iterations=iterations,
        )

    # If public tolerance was reached but tighter polishing exhausted the
    # budget, the public certificate remains valid.
    if official is not None:
        z_out = official["z"]
        return ForwardReceipt(
            accepted=True,
            status="accepted",
            graph=graph,
            params=params,
            candidate_state=z_out,
            official_state=z_out,
            gate=official["gate"],
            residual=float(official["res"]),
            residual_bound=float(official["bound"]),
            actions_used=actions_used,
            max_actions=max_actions,
            reason=None,
            receipt_id=receipt_id,
            mapped_warm_node_ids=mapped,
            unseen_node_ids=unseen,
            support=support,
            iterations=iterations,
        )

    assert best is not None
    return ForwardReceipt(
        accepted=False,
        status="rejected",
        reason=reason,
        graph=graph,
        params=params,
        candidate_state=best["z"],
        residual=float(best["res"]),
        residual_bound=float(best["bound"]),
        actions_used=actions_used,
        max_actions=max_actions,
        receipt_id=receipt_id,
        mapped_warm_node_ids=mapped,
        unseen_node_ids=unseen,
        support=support,
        iterations=iterations,
    )
