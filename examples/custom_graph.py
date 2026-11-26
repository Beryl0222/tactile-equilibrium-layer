#!/usr/bin/env python3
"""Small replaceable graph/parameter example.

Run from the repository root:
    python examples/custom_graph.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from equilibrium_layer import (  # noqa: E402
    EquilibriumSession,
    Graph,
    GraphFamily,
    Parameters,
)


def main() -> int:
    graph = Graph.from_arrays(
        node_ids=("contact-a", "contact-b", "new-pad"),
        X=np.array([[0.4, -0.1], [0.2, 0.3], [0.0, 0.1]]),
        edges=[(0, 1, 0.25), (1, 0, 0.2), (0, 0, 0.1)],
        sample_weight=1.5,
    )
    params = Parameters(
        W=0.2 * np.eye(3),
        U=np.array([[0.1, -0.05, 0.02], [0.0, 0.08, -0.03]]),
        b=np.zeros(3),
    )
    family = GraphFamily("tiny-family", max_abs_row_sum=0.6, rho=0.92)
    session = EquilibriumSession(params, [family])
    window = session.begin_window(window_id="example")

    forward = session.solve(graph, max_actions=100)
    print("forward:", forward.status, "viewable=", forward.viewable,
          "residual=", forward.residual)
    print("accepted nodes:", forward.accepted_node_ids)

    gradient = session.backward(
        forward,
        {"contact-a": np.array([0.3, -0.2, 0.1])},
        max_actions=100,
        inverse_risk_limit=100.0,
    )
    print("backward:", gradient.status, "trainable=", gradient.trainable,
          "relative_residual=", gradient.relative_residual)
    window.add(gradient)

    candidate = params.next_generation(W=0.25 * np.eye(3))
    update = session.propose_update(candidate, window)
    print("update:", update.status, "accepted=", update.accepted,
          "generation=", session.generation)
    return 0 if forward.viewable and gradient.trainable and update.accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
