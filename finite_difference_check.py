#!/usr/bin/env python3
"""Print dense-oracle and central finite-difference comparisons for materials."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np

from equilibrium_layer import Graph, Parameters, implicit_gradients, solve_equilibrium
from equilibrium_layer.numerical import (
    compare_gradients,
    direct_dense_implicit_gradients,
    finite_difference_gradients,
)


def main() -> int:
    materials = json.loads((ROOT / "data/materials.json").read_text(encoding="utf-8"))
    params = Parameters.from_mapping(materials["parameters"])
    report: dict[str, object] = {}
    all_ok = True

    for index, graph_data in enumerate(materials["graphs"]):
        gid = graph_data["id"]
        graph = Graph.from_mapping(graph_data)
        upstream = np.asarray(materials["upstream"][index])
        forward = solve_equilibrium(graph, params, max_actions=300)
        receipt = implicit_gradients(
            forward, upstream, max_actions=300, inverse_risk_limit=1000
        )
        dx_dense, grads_dense, _, dense_res = direct_dense_implicit_gradients(forward, upstream)
        fd_x, fd_grads = finite_difference_gradients(
            forward, upstream, eps=1e-5, max_actions=800, check_x=True
        )
        comparison = compare_gradients(
            receipt.gradients,
            fd_grads,
            analytical_x=receipt.grad_X,
            fd_x=fd_x,
        )
        dense_diff = grads_dense.relative_difference(receipt.gradients)
        report[gid] = {
            "forward_accepted": forward.accepted,
            "forward_residual": forward.residual,
            "backward_eligible": receipt.eligible,
            "linear_relative_residual": receipt.relative_residual,
            "dense_oracle_residual": dense_res,
            "matrix_free_vs_dense_max_relative_error": dense_diff["max"],
            "finite_difference": comparison,
        }
        all_ok &= comparison["ok"] and receipt.eligible

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not all_ok:
        return 1
    print("\nFinite-difference and dense implicit-gradient checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
