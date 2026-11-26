"""Replay the ordered materials/events input through the real layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .family import GraphFamily
from .graph import Graph
from .parameters import Parameters
from .session import EquilibriumSession


def run_event_stream(materials: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    params = Parameters.from_mapping(materials["parameters"], generation=0)
    families = {
        f["id"]: GraphFamily.from_mapping(f) for f in materials.get("families", [])
    }
    session = EquilibriumSession(params, families)
    window = session.begin_window(window_id="materials-window")

    graphs = {g["id"]: Graph.from_mapping(g) for g in materials["graphs"]}
    revisions: dict[tuple[str, int], Graph] = {}
    gr = materials.get("graph_revision")
    if gr:
        revisions[(gr["id"], int(gr["revision"]))] = Graph.from_mapping(gr)

    upstream_by_graph = {
        g["id"]: materials["upstream"][i] for i, g in enumerate(materials["graphs"])
    }
    candidates = {c["id"]: c for c in materials.get("candidate_parameters", [])}

    handles: dict[str, Any] = {}
    trace: list[dict[str, Any]] = []
    updates: list[dict[str, Any]] = []

    for event in events:
        op = event["op"]
        if op == "forward":
            gid = event["graph"]
            rev = int(event.get("revision", graphs[gid].revision))
            graph = revisions.get((gid, rev), graphs[gid])
            handle = session.solve(graph, max_actions=int(event.get("max_actions", 100)))
            handles[gid] = handle
            trace.append({"op": op, **handle.public_dict()})

        elif op == "revise_graph":
            gid = event["graph"]
            rev = int(event["revision"])
            graph = revisions.get((gid, rev))
            if graph is None:
                raise KeyError(f"no supplied revision {gid} r{rev}")
            graphs[gid] = graph
            trace.append({
                "op": op,
                "graph_id": gid,
                "revision": rev,
                "fingerprint": graph.fingerprint,
                "note": "revision registered; array positions are not node identities",
            })

        elif op == "backward":
            gid = event["graph"]
            handle = handles[gid]
            receipt = session.backward(
                handle,
                upstream_by_graph.get(gid),
                max_actions=int(event.get("max_actions", 100)),
                inverse_risk_limit=float(event.get("inverse_risk_limit", 100.0)),
            )
            attribution = window.add(receipt)
            item = {
                "op": op,
                "attribution_accepted": attribution.accepted,
                "attribution_reason": attribution.reason,
                **receipt.public_dict(),
            }
            trace.append(item)

        elif op == "propose_update":
            cid = event["candidate"]
            result = session.propose_update(candidates[cid], window)
            updates.append({"candidate": cid, **result.public_dict()})
            trace.append({"op": op, "candidate": cid, **result.public_dict()})

        else:
            raise ValueError(f"unknown event op {op!r}")

    return {
        "final_parameter_generation": session.generation,
        "final_W": session.params.W.tolist(),
        "accepted_warm_node_ids": {k: list(v) for k, v in session.accepted_state_summary().items()},
        "effective_window_weight": window.effective_weight,
        "eligible_window_samples": list(window.eligible_sample_ids),
        "memory": session.memory_report(),
        "updates": updates,
        "trace": trace,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run materials/events through the equilibrium layer")
    parser.add_argument("--materials", default="data/materials.json")
    parser.add_argument("--events", default="data/events.json")
    parser.add_argument("--output", default=None, help="optional JSON output path")
    args = parser.parse_args(argv)

    materials = json.loads(Path(args.materials).read_text(encoding="utf-8"))
    events = json.loads(Path(args.events).read_text(encoding="utf-8"))
    result = run_event_stream(materials, events)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
