"""Qualification and weighted aggregation for one learning window."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .backward import GradientReceipt
from .compat import DictAccessMixin
from .family import StabilityCertificate
from .gradients import ParameterGradients


@dataclass(frozen=True)
class WindowAttribution(DictAccessMixin):
    accepted: bool
    sample_id: str
    reason: str | None = None

    def __bool__(self) -> bool:
        return self.accepted


@dataclass
class UpdateResult(DictAccessMixin):
    accepted: bool
    status: str
    old_generation: int
    new_generation: int | None = None
    reason: str | None = None
    reasons: tuple[str, ...] = ()
    aggregate: ParameterGradients | None = None
    effective_sample_weight: float = 0.0
    eligible_sample_ids: tuple[str, ...] = ()
    rejected_attributions: tuple[dict[str, Any], ...] = ()
    stability: StabilityCertificate | None = None
    candidate_fingerprint: str | None = None
    parameters: Any = None
    window_id: str | None = None
    membership_problems: tuple[dict[str, Any], ...] = ()

    @property
    def update_accepted(self) -> bool:
        return self.accepted

    @property
    def success(self) -> bool:
        return self.accepted

    @property
    def stable(self) -> bool | None:
        return None if self.stability is None else self.stability.stable

    @property
    def denominator(self) -> float:
        return self.effective_sample_weight

    def public_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "status": self.status,
            "reason": self.reason,
            "reasons": list(self.reasons),
            "old_generation": self.old_generation,
            "new_generation": self.new_generation,
            "effective_sample_weight": self.effective_sample_weight,
            "eligible_sample_ids": list(self.eligible_sample_ids),
            "rejected_attributions": list(self.rejected_attributions),
            "stability": None if self.stability is None else {
                "status": self.stability.status,
                "stable": self.stability.stable,
                "bound": self.stability.upper_lipschitz_bound,
                "rho": self.stability.rho,
                "proof": self.stability.proof,
            },
            "window_id": self.window_id,
            "membership_problems": list(self.membership_problems),
        }


def aggregate_gradients(
    receipts: "list[GradientReceipt]",
) -> tuple[ParameterGradients | None, float, tuple[str, ...]]:
    """Weighted aggregation of already-qualified receipts only.

    Rejected receipts are ignored completely: they contribute neither a zero
    numerator term nor a denominator slot.
    """

    eligible = [
        r for r in receipts
        if isinstance(r, GradientReceipt) and r.eligible and r.gradients is not None
    ]
    denominator = float(sum(r.sample_weight for r in eligible))
    ids = tuple((r.sample_id or r.graph_id) for r in eligible)
    if denominator <= 0.0:
        return None, denominator, ids
    total: ParameterGradients | None = None
    for receipt in eligible:
        assert receipt.gradients is not None
        term = receipt.gradients * receipt.sample_weight
        total = term if total is None else total + term
    assert total is not None
    return total / denominator, denominator, ids


@dataclass
class LearningWindow:
    """Aggregates only attributed, trainable samples.

    Rejected gradients are retained for diagnosis but never enter the
    numerator or denominator.  Zero-filled rejected gradients therefore cannot
    dilute an accepted batch.
    """

    parameter_generation: int
    parameter_fingerprint: str | None = None
    window_id: str | None = None
    cohort_id: str | None = None
    eligible_receipts: list[GradientReceipt] = field(default_factory=list)
    rejected_receipts: list[tuple[GradientReceipt | None, str]] = field(default_factory=list)
    attribution_decisions: list[WindowAttribution] = field(default_factory=list)
    graph_versions: dict[str, str] = field(default_factory=dict)
    seen_samples: set[str] = field(default_factory=set)
    committed: bool = False
    committed_generation: int | None = None

    def add(self, receipt: GradientReceipt) -> WindowAttribution:
        if not isinstance(receipt, GradientReceipt):
            raise TypeError("LearningWindow.add expects a GradientReceipt")

        sample_id = receipt.sample_id or receipt.graph.graph_id

        def reject(reason: str) -> WindowAttribution:
            decision = WindowAttribution(False, sample_id, reason)
            self.rejected_receipts.append((receipt, reason))
            self.attribution_decisions.append(decision)
            return decision

        if self.committed:
            return reject("window_already_committed")
        if not receipt.eligible or receipt.gradients is None:
            return reject("sample_gradient_not_eligible")
        if receipt.parameter_generation != self.parameter_generation:
            return reject("parameter_generation_mismatch")
        if (
            self.parameter_fingerprint is not None
            and receipt.parameter_fingerprint != self.parameter_fingerprint
        ):
            return reject("parameter_fingerprint_mismatch")
        if self.cohort_id is not None and receipt.cohort_id != self.cohort_id:
            return reject("cohort_mismatch")

        prior_fp = self.graph_versions.get(receipt.graph_id)
        if prior_fp is not None and prior_fp != receipt.graph_fingerprint:
            return reject("mixed_graph_revisions")

        if sample_id in self.seen_samples:
            return reject("duplicate_sample_in_window")

        # Same graph id may occur once in a coherent window; record its exact
        # revision/fingerprint even before duplicate check ordering.
        self.graph_versions[receipt.graph_id] = receipt.graph_fingerprint
        self.seen_samples.add(sample_id)
        self.eligible_receipts.append(receipt)
        decision = WindowAttribution(True, sample_id)
        self.attribution_decisions.append(decision)
        return decision

    @property
    def eligible_count(self) -> int:
        return len(self.eligible_receipts)

    @property
    def effective_weight(self) -> float:
        return float(sum(r.effective_weight for r in self.eligible_receipts))

    # Compatibility with wording in the specification.
    @property
    def effective_sample_weight(self) -> float:
        return self.effective_weight

    @property
    def denominator(self) -> float:
        return self.effective_weight

    @property
    def eligible_sample_ids(self) -> tuple[str, ...]:
        return tuple((r.sample_id or r.graph_id) for r in self.eligible_receipts)

    def aggregate(self) -> ParameterGradients | None:
        denominator = self.effective_weight
        if denominator <= 0.0:
            return None
        total: ParameterGradients | None = None
        for receipt in self.eligible_receipts:
            if receipt.gradients is None:
                continue
            term = receipt.gradients * receipt.sample_weight
            total = term if total is None else total + term
        if total is None:
            return None
        return total / denominator

    def rejected_attribution_dicts(self) -> tuple[dict[str, Any], ...]:
        out: list[dict[str, Any]] = []
        for receipt, reason in self.rejected_receipts:
            out.append(
                {
                    "sample_id": None if receipt is None else (receipt.sample_id or receipt.graph_id),
                    "reason": reason,
                    "parameter_generation": None if receipt is None else receipt.parameter_generation,
                    "graph_fingerprint": None if receipt is None else receipt.graph_fingerprint,
                }
            )
        return tuple(out)


UpdateDecision = UpdateResult
