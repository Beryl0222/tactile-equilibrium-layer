"""Declared graph families and whole-family stability evidence."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from .compat import DictAccessMixin
from .errors import ValidationError
from .graph import SUPPORT
from .parameters import Parameters

CERT_EPS = 1e-12


def lipschitz_upper_bound(W: Any, max_abs_row_sum: float) -> float:
    """Return ``R * ||W||_2`` for the max-row Euclidean state norm."""
    matrix = W.W if hasattr(W, "W") else np.asarray(W, dtype=np.float64)
    return float(max_abs_row_sum * np.linalg.norm(matrix, ord=2))


@dataclass(frozen=True)
class GraphFamily:
    family_id: str
    max_abs_row_sum: float
    rho: float = 0.92
    weight_min: float | None = -1.0
    weight_max: float | None = 1.0
    max_nodes: int = 2048
    max_edges_per_node: int = 12

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "GraphFamily":
        return cls(
            family_id=str(data.get("id", data.get("family_id", "family"))),
            max_abs_row_sum=float(data["max_abs_row_sum"]),
            rho=float(data.get("rho", 0.92)),
            weight_min=None if data.get("weight_min") is None else float(data["weight_min"]),
            weight_max=None if data.get("weight_max") is None else float(data["weight_max"]),
            max_nodes=int(data.get("max_nodes", SUPPORT["max_nodes"])),
            max_edges_per_node=int(data.get("max_edges_per_node", SUPPORT["max_edges_per_node"])),
        )

    @property
    def max_abs_weight(self) -> float | None:
        if self.weight_min is None and self.weight_max is None:
            return None
        lo = -math.inf if self.weight_min is None else abs(float(self.weight_min))
        hi = -math.inf if self.weight_max is None else abs(float(self.weight_max))
        return max(lo, hi)


@dataclass(frozen=True)
class StabilityCertificate(DictAccessMixin):
    status: str  # certified | violated | insufficient_evidence
    stable: bool
    family_id: str | None
    rho: float | None
    sigma_w: float | None
    upper_lipschitz_bound: float | None
    realizable_lower_bound: float | None
    reasons: tuple[str, ...]
    proof: str

    @property
    def certifiable(self) -> bool:
        return self.status == "certified"


def _conservative_le(value: float, limit: float) -> bool:
    # A one-ULP-ish numerical allowance only; callers needing a strict formal
    # interval proof can replace this hook without changing the update flow.
    return value <= limit + 1e-11


def check_graph_membership(
    graph: Any,
    family: GraphFamily,
    *,
    tolerance: float = 1e-12,
) -> "tuple[bool, tuple[str, ...]]":
    """Check actual graph limits against one declared family envelope."""

    reasons: list[str] = []
    node_count = int(getattr(graph, "original_node_count", None) or graph.num_nodes)
    if node_count > family.max_nodes:
        reasons.append("node_count_exceeds_family_max_nodes")
    if graph.max_out_degree > family.max_edges_per_node:
        reasons.append("degree_exceeds_family_limit")
    if graph.num_edges:
        max_weight = float(np.max(np.abs(graph.edge_weight)))
        allowed_weight = family.max_abs_weight
        if allowed_weight is not None and max_weight > allowed_weight + tolerance:
            reasons.append("edge_weight_exceeds_family_range")
    if graph.max_abs_row_sum > family.max_abs_row_sum + 1e-10:
        reasons.append("row_abs_sum_exceeds_family_declaration")
    return not reasons, tuple(reasons)


def certify_stability(
    params: Parameters,
    family: GraphFamily | Mapping[str, Any] | None = None,
    *,
    families: Mapping[str, GraphFamily | Mapping[str, Any]] | None = None,
    family_id: str | None = None,
) -> StabilityCertificate:
    """Certify the max-row Euclidean induced Lipschitz bound.

    For ``F(Z)=tanh(AZW+XU+b)`` and ``||Z||_{2,infty}=max_i ||Z_i||_2``,
    ``Lip(F) <= max_i sum_j |A_ij| ||W||_2 <= R ||W||_2``.  The tanh derivative
    is bounded by one.  This is a whole-family proof from ``R``; convergence of
    a few graphs is never used as the family certificate.
    """

    if family is None and family_id is not None and families is not None:
        family = families.get(family_id)
    if family is None and families is not None and len(families) == 1:
        family = next(iter(families.values()))
    if isinstance(family, Mapping):
        family = GraphFamily.from_mapping(family)

    if family is None:
        return StabilityCertificate(
            status="insufficient_evidence",
            stable=False,
            family_id=family_id,
            rho=None,
            sigma_w=None,
            upper_lipschitz_bound=None,
            realizable_lower_bound=None,
            reasons=("no_declared_family",),
            proof="No graph-family stability declaration was available.",
        )

    if not isinstance(family, GraphFamily):
        raise ValidationError("family must be a GraphFamily or mapping")

    r = float(family.max_abs_row_sum)
    rho = float(family.rho)
    if not math.isfinite(r) or r < 0.0:
        return StabilityCertificate(
            "insufficient_evidence", False, family.family_id, rho, None, None, None,
            ("invalid_max_abs_row_sum",), "Family row-sum declaration is invalid.",
        )
    if not math.isfinite(rho) or rho <= 0.0:
        return StabilityCertificate(
            "insufficient_evidence", False, family.family_id, rho, None, None, None,
            ("invalid_rho",), "Family rho declaration is invalid.",
        )

    W = params.W if isinstance(params, Parameters) else np.asarray(params, dtype=np.float64)
    sigma_w = float(np.linalg.norm(W, ord=2))
    upper = r * sigma_w

    w_abs_max = family.max_abs_weight
    lower: float | None = None
    if w_abs_max is not None and math.isfinite(w_abs_max) and w_abs_max > 0.0:
        realizable_row_sum = min(r, float(family.max_edges_per_node) * w_abs_max)
        if family.max_nodes >= 1:
            lower = realizable_row_sum * sigma_w

    if _conservative_le(upper, rho):
        return StabilityCertificate(
            status="certified",
            stable=True,
            family_id=family.family_id,
            rho=rho,
            sigma_w=sigma_w,
            upper_lipschitz_bound=upper,
            realizable_lower_bound=lower,
            reasons=(),
            proof=(
                f"max_i row_abs_sum(A) <= {r:.12g}; ||W||2={sigma_w:.12g}; "
                f"Lip_2,infty(F) <= {upper:.12g} <= rho={rho:.12g}."
            ),
        )

    if lower is not None and lower > rho + 1e-11:
        return StabilityCertificate(
            status="violated",
            stable=False,
            family_id=family.family_id,
            rho=rho,
            sigma_w=sigma_w,
            upper_lipschitz_bound=upper,
            realizable_lower_bound=lower,
            reasons=("realizable_family_member_exceeds_rho", "upper_bound_exceeds_rho"),
            proof=(
                f"An allowed row can realize row sum {min(r, family.max_edges_per_node * w_abs_max):.12g}; "
                f"at Z=0 the Jacobian lower bound is {lower:.12g} > rho={rho:.12g}."
            ),
        )

    return StabilityCertificate(
        status="insufficient_evidence",
        stable=False,
        family_id=family.family_id,
        rho=rho,
        sigma_w=sigma_w,
        upper_lipschitz_bound=upper,
        realizable_lower_bound=lower,
        reasons=("conservative_proof_exceeds_rho_but_violation_not_established",),
        proof=(
            f"Generic upper bound {upper:.12g} exceeds rho={rho:.12g}; available declarations do not "
            "provide a tighter certifiable proof."
        ),
    )
