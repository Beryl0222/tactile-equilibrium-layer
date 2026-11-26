#!/usr/bin/env python3
"""Run a generated capacity directory through sparse forward/backward paths.

Example:
    python make_capacity.py --output /tmp/product-capacity
    python run_capacity.py --input /tmp/product-capacity --check-backward
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np

from equilibrium_layer import Graph, Parameters, implicit_gradients, solve_equilibrium


def run_capacity(directory: Path, *, check_backward: bool, max_actions: int | None,
                 inverse_risk_limit: float) -> dict:
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    arrays = np.load(directory / "arrays.npz")

    graph = Graph.from_arrays(
        node_ids=metadata["node_ids"],
        X=arrays["X"],
        edge_u=arrays["edge_u"],
        edge_v=arrays["edge_v"],
        edge_weight=arrays["edge_weight"],
        graph_id=metadata.get("graph_id", "capacity-graph"),
        revision=int(metadata.get("revision", 1)),
        sample_weight=float(metadata.get("sample_weight", 1.0)),
    )
    params = Parameters(W=arrays["W"], U=arrays["U"], b=arrays["b"])
    budget = int(max_actions if max_actions is not None else metadata.get("max_actions", 200))

    forward = solve_equilibrium(graph, params, max_actions=budget)
    result = {
        "graph": graph.summary(),
        "forward": forward.public_dict(),
        "parameter_state_dim": params.state_dim,
    }

    if check_backward:
        rng = np.random.default_rng(int(metadata.get("seed", 261126)) + 1)
        upstream = rng.normal(scale=0.1, size=(graph.num_nodes, params.state_dim))
        gradient = implicit_gradients(
            forward,
            upstream,
            max_actions=budget,
            inverse_risk_limit=inverse_risk_limit,
        )
        result["backward"] = gradient.public_dict()
        result["gradient_max_abs"] = (
            None if gradient.gradients is None else float(gradient.gradients.max_abs())
        )

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--check-backward", action="store_true")
    parser.add_argument("--max-actions", type=int, default=None)
    parser.add_argument("--inverse-risk-limit", type=float, default=1000.0)
    args = parser.parse_args()
    result = run_capacity(
        args.input,
        check_backward=args.check_backward,
        max_actions=args.max_actions,
        inverse_risk_limit=args.inverse_risk_limit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
