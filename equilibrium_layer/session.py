"""Stateful session: generations, named warm starts, forward closing, updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import itertools

import numpy as np

from .backward import GradientReceipt, implicit_gradients
from .errors import ClosedForwardError, ValidationError
from .family import GraphFamily, certify_stability, check_graph_membership
from .forward import ForwardReceipt, solve_equilibrium
from .graph import Graph, WarmStart
from .parameters import Parameters
from .window import LearningWindow, UpdateResult

MIB = 1024**2


class ForwardHandle:
    """Caller-facing forward token.

    Attribute access delegates read-only result fields to the underlying
    :class:`ForwardReceipt`.  Closing a handle permanently disables adjoints
    for that forward in its session.
    """

    def __init__(self, session: "EquilibriumSession", receipt: ForwardReceipt) -> None:
        self._session = session
        self._receipt = receipt
        self._closed = False

    @property
    def receipt(self) -> ForwardReceipt:
        return self._receipt

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def receipt_id(self) -> str | None:
        return self._receipt.receipt_id

    def close(self) -> None:
        self._closed = True
        self._session._close_record(self._receipt.receipt_id)

    def backward(self, upstream: Any = None, **kwargs: Any) -> GradientReceipt:
        return self._session.backward(self, upstream, **kwargs)

    def try_backward(self, upstream: Any = None, **kwargs: Any) -> GradientReceipt:
        return self._session.try_backward(self, upstream, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._receipt, name)

    def public_dict(self) -> dict[str, Any]:
        data = self._receipt.public_dict()
        data["handle_closed"] = self._closed
        return data

    def __getitem__(self, key: str) -> Any:
        return self._receipt[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._receipt.get(key, default)


class EquilibriumSession:
    def __init__(
        self,
        params: Parameters | Mapping[str, Any] | None = None,
        families: Mapping[str, GraphFamily | Mapping[str, Any]] | Sequence[GraphFamily | Mapping[str, Any]] | None = None,
        *,
        parameters: Parameters | Mapping[str, Any] | None = None,
        W: Any = None,
        U: Any = None,
        b: Any = None,
        generation: int = 0,
        family: GraphFamily | Mapping[str, Any] | Sequence[GraphFamily | Mapping[str, Any]] | None = None,
        default_family_id: str | None = None,
        resident_forward_budget_bytes: int = 16 * MIB,
        temporary_workspace_budget_bytes: int = 16 * MIB,
    ) -> None:
        if params is None:
            params = parameters
        if params is None and W is not None:
            if U is None or b is None:
                raise ValidationError("W, U and b must all be supplied")
            params = Parameters(W=W, U=U, b=b, generation=generation)
        if params is None:
            raise ValidationError("EquilibriumSession requires params/parameters or W/U/b")
        if isinstance(params, Mapping):
            params = Parameters.from_mapping(params, generation=int(params.get("generation", generation)))
        if not isinstance(params, Parameters):
            raise ValidationError("params must be Parameters or a mapping")
        self.params = params
        if families is None and family is not None:
            families = family
        if isinstance(families, (GraphFamily, Mapping)) and not isinstance(families, dict):
            # A single GraphFamily dataclass; mappings may be a family dict or
            # a mapping of id -> family and are disambiguated below.
            families = [families]
        if isinstance(families, Mapping) and "max_abs_row_sum" in families:
            families = [GraphFamily.from_mapping(families)]
        self.default_family_id = default_family_id
        self.families: dict[str, GraphFamily] = {}
        if families is not None:
            items = families.items() if isinstance(families, Mapping) else [
                (f.family_id, f) if isinstance(f, GraphFamily) else (f["id"], GraphFamily.from_mapping(f))
                for f in families
            ]
            for key, family in items:
                if isinstance(family, Mapping):
                    family = GraphFamily.from_mapping(family)
                self.families[str(key)] = family
        if default_family_id is None and len(self.families) == 1:
            self.default_family_id = next(iter(self.families))
        self.resident_forward_budget_bytes = int(resident_forward_budget_bytes)
        self.temporary_workspace_budget_bytes = int(temporary_workspace_budget_bytes)
        self.warm_starts: dict[str, WarmStart] = {}
        self.records: dict[str, dict[str, Any]] = {}
        self._ids = itertools.count(1)

    @property
    def generation(self) -> int:
        return self.params.generation

    def set_parameters(
        self,
        params: Parameters | Mapping[str, Any] | None = None,
        *,
        W: Any = None,
        U: Any = None,
        b: Any = None,
        generation: int | None = None,
    ) -> Parameters:
        """Replace shared parameters explicitly (new generation if requested)."""
        if params is None and W is not None:
            params = Parameters(W, U, b, generation=self.generation + 1 if generation is None else generation)
        elif isinstance(params, Mapping):
            params = Parameters.from_mapping(
                params, generation=self.generation + 1 if generation is None else generation
            )
        if not isinstance(params, Parameters):
            raise ValidationError("provide params mapping/object or W/U/b")
        if generation is not None and params.generation != generation:
            params = Parameters(params.W, params.U, params.b, generation=generation)
        self.params = params
        return params

    def accepted_warm_start_for(self, graph: Graph) -> WarmStart | None:
        return self.warm_starts.get(graph.graph_id)

    def solve(
        self,
        graph: Graph | Mapping[str, Any],
        *,
        use_accepted_warm_start: bool = True,
        warm_start: WarmStart | Mapping[Any, np.ndarray] | str | None = "accepted",
        max_actions: int = 100,
        commit_warm_start: bool = True,
        **kwargs: Any,
    ) -> ForwardHandle:
        if isinstance(graph, Mapping):
            graph = Graph.from_mapping(graph)

        chosen_warm: WarmStart | Mapping[Any, np.ndarray] | None
        if isinstance(warm_start, str):
            if warm_start == "accepted":
                chosen_warm = self.accepted_warm_start_for(graph) if use_accepted_warm_start else None
            elif warm_start.lower() in ("none", "zero", "zeros"):
                chosen_warm = None
            else:
                raise ValidationError(f"unknown warm_start directive {warm_start!r}")
        elif warm_start is None:
            chosen_warm = self.accepted_warm_start_for(graph) if use_accepted_warm_start else None
        else:
            chosen_warm = warm_start

        receipt_id = f"forward-{next(self._ids)}"
        receipt = solve_equilibrium(
            graph,
            self.params,
            max_actions=max_actions,
            warm_start=chosen_warm,
            receipt_id=receipt_id,
            **kwargs,
        )
        handle = ForwardHandle(self, receipt)
        self.records[receipt_id] = {"receipt": receipt, "closed": False, "handle": handle}

        # Only a certified equilibrium becomes the official next-frame start.
        if receipt.accepted and commit_warm_start and receipt.official_state is not None:
            self.warm_starts[graph.graph_id] = WarmStart(
                node_ids=graph.node_ids,
                state=receipt.official_state,
                source_graph_id=graph.graph_id,
                source_revision=graph.revision,
                source_fingerprint=graph.fingerprint,
                parameter_generation=receipt.parameter_generation,
            )
        return handle

    def _resolve_forward_id(self, forward: ForwardHandle | ForwardReceipt | str) -> str:
        if isinstance(forward, ForwardHandle):
            return str(forward.receipt_id)
        if isinstance(forward, ForwardReceipt):
            if forward.receipt_id is None:
                raise ValidationError("functional ForwardReceipt is not registered in this session")
            return forward.receipt_id
        return str(forward)

    def _close_record(self, receipt_id: str | None) -> None:
        if receipt_id is not None and receipt_id in self.records:
            self.records[receipt_id]["closed"] = True

    def close_forward(self, forward: ForwardHandle | ForwardReceipt | str) -> None:
        receipt_id = self._resolve_forward_id(forward)
        self._close_record(receipt_id)

    def is_closed(self, forward: ForwardHandle | ForwardReceipt | str) -> bool:
        receipt_id = self._resolve_forward_id(forward)
        return bool(self.records.get(receipt_id, {}).get("closed", False))

    def _closed_gradient_receipt(self, forward: ForwardReceipt) -> GradientReceipt:
        from .backward import LinearSolveReport, jacobian_norm_bounds

        linear = LinearSolveReport(
            False, float("inf"), 1e-6, 0, None, "rejected_closed", "forward_closed", None
        )
        gate = forward.gate if forward.gate is not None else np.zeros(
            (forward.graph.num_nodes, forward.params.state_dim)
        )
        risk = jacobian_norm_bounds(forward.graph, forward.params, gate)
        return GradientReceipt(
            eligible=False,
            status="forward_closed",
            graph=forward.graph,
            params=forward.params,
            forward_receipt_id=forward.receipt_id,
            upstream=np.zeros((forward.graph.num_nodes, forward.params.state_dim)),
            linear=linear,
            risk=risk,
            reasons=("forward_closed_by_caller",),
        )

    def backward(
        self,
        forward: ForwardHandle | ForwardReceipt | str | None = None,
        upstream: Any = None,
        *,
        forward_id: str | None = None,
        raise_if_closed: bool = True,
        **kwargs: Any,
    ) -> GradientReceipt:
        target = forward if forward is not None else forward_id
        if target is None:
            raise ValidationError("backward requires a ForwardHandle, receipt or forward_id")
        receipt_id = self._resolve_forward_id(target)
        record = self.records.get(receipt_id)
        if record is None:
            raise ValidationError(f"unknown or unregistered forward {receipt_id!r}")
        receipt: ForwardReceipt = record["receipt"]
        closed = bool(record["closed"]) or (
            isinstance(target, ForwardHandle) and target.closed
        )
        if closed:
            if raise_if_closed:
                raise ClosedForwardError(f"forward {receipt_id} was explicitly closed")
            return self._closed_gradient_receipt(receipt)

        cohort_id = kwargs.pop("cohort_id", None)
        sample_id = kwargs.pop("sample_id", None)
        return implicit_gradients(
            receipt, upstream, cohort_id=cohort_id, sample_id=sample_id, **kwargs
        )

    def try_backward(self, forward: Any = None, upstream: Any = None, **kwargs: Any) -> GradientReceipt:
        """Backward that returns a ``forward_closed`` receipt instead of raising."""
        kwargs["raise_if_closed"] = False
        return self.backward(forward, upstream, **kwargs)

    def begin_window(
        self,
        *,
        window_id: str | None = None,
        cohort_id: str | None = None,
    ) -> LearningWindow:
        return LearningWindow(
            parameter_generation=self.generation,
            parameter_fingerprint=self.params.fingerprint,
            window_id=window_id or f"window-{next(self._ids)}",
            cohort_id=cohort_id,
        )

    def _select_family(self, window: LearningWindow, family_id: str | None) -> GraphFamily | None:
        key = family_id or self.default_family_id
        if key is not None:
            return self.families.get(str(key))
        candidate_ids = {r.graph.family_id for r in window.eligible_receipts if r.graph.family_id}
        if len(candidate_ids) == 1:
            fid = next(iter(candidate_ids))
            return self.families.get(str(fid))
        if len(self.families) == 1:
            return next(iter(self.families.values()))
        return None

    def propose_update(
        self,
        candidate: Parameters | Mapping[str, Any],
        window: LearningWindow,
        *,
        family_id: str | None = None,
        commit: bool = True,
    ) -> UpdateResult:
        old_generation = self.generation
        rejected_attributions = window.rejected_attribution_dicts()

        if not isinstance(window, LearningWindow):
            raise ValidationError("window must be a LearningWindow from session.begin_window")
        if window.committed:
            return UpdateResult(
                False, "window_already_committed", old_generation,
                reason="window_already_committed",
                reasons=("window_already_committed",),
                effective_sample_weight=window.effective_weight,
                eligible_sample_ids=window.eligible_sample_ids,
                rejected_attributions=rejected_attributions,
                parameters=self.params,
                window_id=window.window_id,
            )
        if window.parameter_generation != self.generation or (
            window.parameter_fingerprint is not None
            and window.parameter_fingerprint != self.params.fingerprint
        ):
            return UpdateResult(
                False, "generation_mismatch", old_generation,
                reason="parameter_generation_mismatch",
                reasons=("parameter_generation_mismatch",),
                effective_sample_weight=window.effective_weight,
                eligible_sample_ids=window.eligible_sample_ids,
                rejected_attributions=rejected_attributions,
                parameters=self.params,
                window_id=window.window_id,
            )

        aggregate = window.aggregate()
        if aggregate is None or window.effective_weight <= 0.0:
            return UpdateResult(
                False, "no_eligible_gradient", old_generation,
                reason="no_eligible_gradient",
                reasons=("all_samples_rejected_or_zero_denominator",),
                aggregate=None,
                effective_sample_weight=window.effective_weight,
                eligible_sample_ids=window.eligible_sample_ids,
                rejected_attributions=rejected_attributions,
                parameters=self.params,
                window_id=window.window_id,
            )

        if isinstance(candidate, Mapping):
            candidate_params = Parameters(
                W=candidate["W"], U=candidate["U"], b=candidate["b"],
                generation=old_generation + 1,
                parameter_id=candidate.get("id") or candidate.get("parameter_id"),
            )
        elif isinstance(candidate, Parameters):
            candidate_params = Parameters(
                W=candidate.W, U=candidate.U, b=candidate.b,
                generation=old_generation + 1,
                parameter_id=candidate.parameter_id,
            )
        else:
            raise ValidationError("candidate must be Parameters or a mapping")

        family = self._select_family(window, family_id)
        certificate = certify_stability(candidate_params, family)
        if certificate.status != "certified":
            status = (
                "stability_violated"
                if certificate.status == "violated"
                else "stability_evidence_insufficient"
            )
            return UpdateResult(
                False, status, old_generation,
                reason=certificate.status,
                reasons=certificate.reasons or (certificate.status,),
                aggregate=aggregate,
                effective_sample_weight=window.effective_weight,
                eligible_sample_ids=window.eligible_sample_ids,
                rejected_attributions=rejected_attributions,
                stability=certificate,
                candidate_fingerprint=candidate_params.fingerprint,
                parameters=self.params,
                window_id=window.window_id,
            )

        membership_problems: list[dict[str, Any]] = []
        if family is not None:
            for receipt in window.eligible_receipts:
                member_ok, member_reasons = check_graph_membership(receipt.graph, family)
                if not member_ok:
                    membership_problems.append({
                        "sample_id": receipt.sample_id or receipt.graph_id,
                        "graph_id": receipt.graph_id,
                        "revision": receipt.graph_revision,
                        "reasons": list(member_reasons),
                    })
        if membership_problems:
            return UpdateResult(
                False, "graph_outside_declared_family", old_generation,
                reason="graph_outside_declared_family",
                reasons=("gradient_graph_not_in_selected_family",),
                aggregate=aggregate,
                effective_sample_weight=window.effective_weight,
                eligible_sample_ids=window.eligible_sample_ids,
                rejected_attributions=rejected_attributions,
                stability=certificate,
                candidate_fingerprint=candidate_params.fingerprint,
                parameters=self.params,
                window_id=window.window_id,
                membership_problems=tuple(membership_problems),
            )

        if not commit:
            return UpdateResult(
                accepted=True,
                status="qualified_preview",
                old_generation=old_generation,
                new_generation=candidate_params.generation,
                reason=None,
                reasons=(),
                aggregate=aggregate,
                effective_sample_weight=window.effective_weight,
                eligible_sample_ids=window.eligible_sample_ids,
                rejected_attributions=rejected_attributions,
                stability=certificate,
                candidate_fingerprint=candidate_params.fingerprint,
                parameters=candidate_params,
                window_id=window.window_id,
            )

        # Whole transaction commits only here.
        self.params = candidate_params
        window.committed = True
        window.committed_generation = candidate_params.generation
        return UpdateResult(
            accepted=True,
            status="accepted",
            old_generation=old_generation,
            new_generation=candidate_params.generation,
            reason=None,
            reasons=(),
            aggregate=aggregate,
            effective_sample_weight=window.effective_weight,
            eligible_sample_ids=window.eligible_sample_ids,
            rejected_attributions=rejected_attributions,
            stability=certificate,
            candidate_fingerprint=candidate_params.fingerprint,
            parameters=self.params,
            window_id=window.window_id,
        )

    def check_update(
        self,
        candidate: Parameters | Mapping[str, Any],
        window: LearningWindow,
        *,
        family_id: str | None = None,
    ) -> UpdateResult:
        """Dry-run update qualification without changing parameters/window."""
        return self.propose_update(candidate, window, family_id=family_id, commit=False)

    def memory_report(self) -> dict[str, int | bool]:
        resident = 0
        graph_copies = 0
        active_records = 0
        for record in self.records.values():
            if not record["closed"]:
                receipt = record["receipt"]
                resident += receipt.retained_bytes + receipt.graph.storage_bytes
                graph_copies += receipt.graph.storage_bytes
                active_records += 1
        return {
            "resident_forward_bytes_estimate": resident,
            "state_checkpoint_bytes_estimate": resident - graph_copies,
            "copied_graph_array_bytes": graph_copies,
            "resident_budget_bytes": self.resident_forward_budget_bytes,
            "resident_within_budget": resident <= self.resident_forward_budget_bytes,
            "active_forward_records": active_records,
            "temporary_workspace_budget_bytes": self.temporary_workspace_budget_bytes,
        }

    def accepted_state_summary(self) -> dict[str, tuple[Any, ...]]:
        return {gid: tuple(ws.node_ids) for gid, ws in self.warm_starts.items()}


# Discoverable method aliases for callers using forward/update vocabulary.
EquilibriumSession.forward = EquilibriumSession.solve  # type: ignore[attr-defined]
EquilibriumSession.submit_update = EquilibriumSession.propose_update  # type: ignore[attr-defined]
EquilibriumSession.update = EquilibriumSession.propose_update  # type: ignore[attr-defined]
