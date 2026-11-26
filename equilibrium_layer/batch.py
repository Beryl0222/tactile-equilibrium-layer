"""Convenience batch orchestration with independent sample qualification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .backward import GradientReceipt
from .graph import Graph
from .session import EquilibriumSession, ForwardHandle
from .window import LearningWindow, WindowAttribution


@dataclass
class SampleDecision:
    sample_id: str
    forward: ForwardHandle
    gradient: GradientReceipt | None = None
    attribution: WindowAttribution | None = None

    @property
    def viewable(self) -> bool:
        return bool(self.forward.viewable)

    @property
    def forward_accepted(self) -> bool:
        return bool(self.forward.accepted)

    @property
    def trainable(self) -> bool:
        return bool(self.gradient is not None and self.gradient.eligible)

    @property
    def in_window(self) -> bool:
        return bool(self.attribution and self.attribution.accepted)

    def rejection_stage(self) -> str | None:
        if not self.forward_accepted:
            return "solve"
        if self.gradient is None or not self.gradient.eligible:
            return "gradient"
        if self.attribution is not None and not self.attribution.accepted:
            return "window_attribution"
        return None

    def public_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "viewable": self.viewable,
            "trainable": self.trainable,
            "in_window": self.in_window,
            "rejection_stage": self.rejection_stage(),
            "forward": self.forward.public_dict(),
            "gradient": None if self.gradient is None else self.gradient.public_dict(),
            "attribution": None if self.attribution is None else {
                "accepted": self.attribution.accepted,
                "reason": self.attribution.reason,
            },
        }


@dataclass
class BatchReport:
    session: EquilibriumSession
    window: LearningWindow | None
    decisions: list[SampleDecision] = field(default_factory=list)

    @property
    def viewable_sample_ids(self) -> tuple[str, ...]:
        return tuple(d.sample_id for d in self.decisions if d.viewable)

    @property
    def trainable_sample_ids(self) -> tuple[str, ...]:
        return tuple(d.sample_id for d in self.decisions if d.trainable)

    @property
    def solve_rejected_ids(self) -> tuple[str, ...]:
        return tuple(d.sample_id for d in self.decisions if d.rejection_stage() == "solve")

    @property
    def gradient_rejected_ids(self) -> tuple[str, ...]:
        return tuple(d.sample_id for d in self.decisions if d.rejection_stage() == "gradient")

    def public_dict(self) -> dict[str, Any]:
        return {
            "viewable_sample_ids": list(self.viewable_sample_ids),
            "trainable_sample_ids": list(self.trainable_sample_ids),
            "solve_rejected_ids": list(self.solve_rejected_ids),
            "gradient_rejected_ids": list(self.gradient_rejected_ids),
            "effective_window_weight": None if self.window is None else self.window.effective_weight,
            "decisions": [d.public_dict() for d in self.decisions],
            "accepted_warm_node_ids": {
                k: list(v) for k, v in self.session.accepted_state_summary().items()
            },
            "memory": self.session.memory_report(),
        }


def process_batch(
    session: EquilibriumSession,
    samples: Sequence[Graph | Mapping[str, Any] | tuple[Any, Any] | Mapping[str, Any]],
    *,
    window: LearningWindow | None = None,
    forward_options: Mapping[str, Any] | None = None,
    backward_options: Mapping[str, Any] | None = None,
    add_to_window: bool = True,
) -> BatchReport:
    """Solve and optionally differentiate every sample independently.

    A failed solve or gradient is recorded on that :class:`SampleDecision`;
    it never aborts the remaining samples.
    """

    forward_options = dict(forward_options or {})
    backward_options = dict(backward_options or {})
    decisions: list[SampleDecision] = []

    for index, sample in enumerate(samples):
        upstream: Any = None
        specified_nodes: Any = None
        sample_id: str | None = None
        opts: dict[str, Any] = {}
        bopts: dict[str, Any] = {}

        if isinstance(sample, tuple) and len(sample) == 2:
            graph_like, upstream = sample
        elif isinstance(sample, Mapping) and "graph" in sample:
            graph_like = sample["graph"]
            upstream = sample.get("upstream")
            specified_nodes = sample.get("specified_nodes")
            sample_id = sample.get("sample_id")
            opts = sample.get("forward_options") or {}
            bopts = sample.get("backward_options") or {}
        else:
            graph_like = sample
            opts = {}
            bopts = {}

        graph = Graph.from_mapping(graph_like) if isinstance(graph_like, Mapping) else graph_like
        if sample_id is None:
            sample_id = graph.graph_id or f"sample-{index}"

        handle = session.solve(graph, **{**forward_options, **opts})
        decision = SampleDecision(sample_id=sample_id, forward=handle)

        if upstream is not None or specified_nodes is not None:
            kwargs = dict(backward_options)
            kwargs.update(bopts)
            if specified_nodes is not None:
                kwargs["specified_nodes"] = specified_nodes
            receipt = session.backward(handle, upstream, **kwargs)
            receipt.sample_id = sample_id if receipt.sample_id in (None, graph.graph_id) else receipt.sample_id
            decision.gradient = receipt
            if window is not None and add_to_window:
                decision.attribution = window.add(receipt)

        decisions.append(decision)

    return BatchReport(session=session, window=window, decisions=decisions)
