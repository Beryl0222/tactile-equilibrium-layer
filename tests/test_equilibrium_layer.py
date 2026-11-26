"""Behavioral and numerical tests for the qualified equilibrium layer."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from equilibrium_layer import (
    EquilibriumSession,
    Graph,
    GraphFamily,
    LearningWindow,
    Parameters,
    ValidationError,
    WarmStart,
    align_warm_start,
    certify_stability,
    implicit_gradients,
    solve_equilibrium,
)
from equilibrium_layer.errors import ClosedForwardError
from equilibrium_layer.events import run_event_stream
from equilibrium_layer.batch import process_batch
from equilibrium_layer.numerical import (
    compare_gradients,
    direct_dense_implicit_gradients,
    finite_difference_gradients,
)

ROOT = Path(__file__).resolve().parents[1]
MATERIALS = json.loads((ROOT / "data/materials.json").read_text(encoding="utf-8"))
EVENTS = json.loads((ROOT / "data/events.json").read_text(encoding="utf-8"))


def material_params() -> Parameters:
    return Parameters.from_mapping(MATERIALS["parameters"])


def material_family() -> GraphFamily:
    return GraphFamily.from_mapping(MATERIALS["families"][0])


class MaterialEventContractTests(unittest.TestCase):
    def test_ordered_events_have_expected_qualifications(self):
        result = run_event_stream(MATERIALS, EVENTS)

        forwards = [x for x in result["trace"] if x["op"] == "forward"]
        backwards = [x for x in result["trace"] if x["op"] == "backward"]
        updates = result["updates"]

        self.assertEqual(len(forwards), 3)
        self.assertTrue(all(x["accepted"] and x["viewable"] for x in forwards))
        self.assertLessEqual(forwards[0]["residual"], forwards[0]["residual_bound"])
        self.assertLessEqual(forwards[1]["residual"], forwards[1]["residual_bound"])

        self.assertFalse(backwards[0]["eligible"])
        self.assertFalse(backwards[0]["risk"]["risk_ok"])
        self.assertFalse(backwards[0]["linear_residual_ok"])
        self.assertTrue(backwards[1]["eligible"])
        self.assertLessEqual(backwards[1]["linear_residual"], 1e-6)
        self.assertLessEqual(backwards[1]["risk"]["inverse_norm_bound"], 100.0)

        self.assertEqual(result["effective_window_weight"], 2.0)
        self.assertEqual(result["eligible_window_samples"], ["g1"])

        self.assertFalse(updates[0]["accepted"])
        self.assertEqual(updates[0]["status"], "stability_violated")
        self.assertTrue(updates[1]["accepted"])
        self.assertEqual(updates[1]["new_generation"], 1)
        self.assertEqual(result["final_parameter_generation"], 1)
        self.assertAlmostEqual(result["final_W"][0][0], 0.3)

    def test_forward_accepted_but_backward_risk_rejected(self):
        g = Graph.from_mapping(MATERIALS["graphs"][1])
        p = material_params()
        fwd = solve_equilibrium(g, p, max_actions=100)
        grad = implicit_gradients(
            fwd, MATERIALS["upstream"][1], max_actions=100, inverse_risk_limit=1.0
        )
        self.assertTrue(fwd.viewable)
        self.assertTrue(grad.linear_residual_ok)
        self.assertFalse(grad.inverse_risk_ok)
        self.assertFalse(grad.trainable)
        self.assertIsNone(grad.gradients)

    def test_one_action_fails_linear_while_normalized_risk_can_pass(self):
        g = Graph.from_mapping(MATERIALS["graphs"][0])
        p = material_params()
        fwd = solve_equilibrium(g, p, max_actions=100)
        grad = implicit_gradients(
            fwd,
            MATERIALS["upstream"][0],
            max_actions=1,
            inverse_risk_limit=0.1,
            risk_metric="amplification_excess",
        )
        self.assertTrue(fwd.viewable)
        self.assertFalse(grad.linear_residual_ok)
        self.assertTrue(grad.risk.amplification_excess_ok)
        self.assertTrue(grad.risk.condition_excess_ok)
        self.assertFalse(grad.eligible)


class WarmStartIdentityTests(unittest.TestCase):
    def test_maps_names_not_positions(self):
        graph = Graph.from_arrays(
            ["b", "new", "a"],
            np.zeros((3, 2)),
            edges=[],
        )
        old = WarmStart(("a", "b"), np.array([[1.0, 2.0], [3.0, 4.0]]))
        z0, mapped, unseen = align_warm_start(graph, old, 2)
        np.testing.assert_allclose(z0[graph.node_ids.index("a")], [1, 2])
        np.testing.assert_allclose(z0[graph.node_ids.index("b")], [3, 4])
        np.testing.assert_allclose(z0[graph.node_ids.index("new")], [0, 0])
        self.assertEqual(set(mapped), {"a", "b"})
        self.assertEqual(unseen, ("new",))

    def test_bare_positional_array_is_rejected(self):
        g = Graph.from_mapping(MATERIALS["graphs"][0])
        with self.assertRaises(ValidationError):
            solve_equilibrium(g, material_params(), warm_start=np.zeros((6, 3)))

    def test_revision_uses_shared_identities_and_zero_for_replacement(self):
        session = EquilibriumSession(material_params(), [material_family()])
        old_graph = Graph.from_mapping(MATERIALS["graphs"][0])
        old = session.solve(old_graph, max_actions=100)
        self.assertTrue(old.accepted)
        old_warm = session.accepted_warm_start_for(old_graph)

        revised = Graph.from_mapping(MATERIALS["graph_revision"])
        new = session.solve(revised, max_actions=100)
        info = new.public_dict()
        self.assertTrue(new.accepted)
        self.assertEqual(len(info["mapped_warm_nodes"]), 5)
        self.assertEqual(info["new_zero_start_nodes"], ["replacement-node"])
        z0, mapped, unseen = align_warm_start(revised, old_warm, 3)
        self.assertEqual(unseen, ("replacement-node",))
        # The replacement row starts at zero even though array position 5 in
        # the old graph contained n0-5.
        np.testing.assert_array_equal(z0[revised.node_ids.index("replacement-node")], 0.0)
        np.testing.assert_allclose(
            z0[revised.node_ids.index("n0-0")],
            old.official_state[old_graph.node_ids.index("n0-0")],
        )
        warm = session.accepted_warm_start_for(revised)
        self.assertIsNotNone(warm)
        self.assertEqual(warm.source_revision, 2)

    def test_budget_failure_does_not_commit_warm_state(self):
        session = EquilibriumSession(material_params())
        g = Graph.from_mapping(MATERIALS["graphs"][0])
        handle = session.solve(g, max_actions=0)
        self.assertFalse(handle.accepted)
        self.assertNotIn("g0", session.warm_starts)


class InvalidNodesTests(unittest.TestCase):
    def test_invalid_nodes_and_their_edges_are_excluded(self):
        nodes = ["ok0", "bad", "ok1"]
        X = np.ones((3, 2))
        edges = [(0, 1, 100.0), (1, 2, 100.0), (0, 2, 0.2)]
        g = Graph.from_arrays(nodes, X, edges=edges, valid=[True, False, True])
        self.assertEqual(g.node_ids, ("ok0", "ok1"))
        self.assertEqual(g.num_edges, 1)
        np.testing.assert_allclose(g.dense_A(), [[0.0, 0.2], [0.0, 0.0]])

        p = Parameters(W=np.eye(3) * 0.1, U=np.zeros((2, 3)), b=np.zeros(3))
        fwd = solve_equilibrium(g, p, max_actions=50)
        self.assertEqual(fwd.official_state.shape, (2, 3))
        upstream_full = np.arange(9, dtype=float).reshape(3, 3)
        grad = implicit_gradients(
            fwd, upstream_full, max_actions=50, inverse_risk_limit=100
        )
        self.assertTrue(grad.eligible, grad.reasons)
        self.assertEqual(grad.upstream.shape, (2, 3))
        np.testing.assert_allclose(grad.upstream[1], upstream_full[2])
        self.assertEqual(grad.grad_X.shape, (2, 2))
        full_x_grad = grad.grad_X_full
        self.assertEqual(full_x_grad.shape, (3, 2))
        np.testing.assert_array_equal(full_x_grad[1], 0.0)


class FiniteDifferenceTests(unittest.TestCase):
    def test_matrix_free_vjp_matches_dense_and_finite_differences(self):
        g = Graph.from_mapping(MATERIALS["graphs"][1])
        p = material_params()
        upstream = np.asarray(MATERIALS["upstream"][1])
        fwd = solve_equilibrium(g, p, max_actions=200)
        self.assertTrue(fwd.accepted)

        receipt = implicit_gradients(
            fwd, upstream, max_actions=200, inverse_risk_limit=1000
        )
        self.assertTrue(receipt.eligible, receipt.reasons)

        dx_dense, grads_dense, _, dense_res = direct_dense_implicit_gradients(fwd, upstream)
        self.assertLess(dense_res, 1e-10)
        diff = grads_dense.relative_difference(receipt.gradients)
        self.assertLess(diff["max"], 1e-9)
        np.testing.assert_allclose(dx_dense, receipt.grad_X, atol=1e-10)

        fd_x, fd_grads = finite_difference_gradients(
            fwd, upstream, eps=1e-5, max_actions=500, check_x=True
        )
        comparison = compare_gradients(
            receipt.gradients,
            fd_grads,
            analytical_x=receipt.grad_X,
            fd_x=fd_x,
            rtol=3e-4,
            atol=3e-6,
        )
        self.assertTrue(comparison["ok"], comparison)


class WindowAndUpdateTests(unittest.TestCase):
    def setUp(self):
        self.session = EquilibriumSession(material_params(), [material_family()])
        self.g0 = Graph.from_mapping(MATERIALS["graphs"][0])
        self.g1 = Graph.from_mapping(MATERIALS["graphs"][1])
        self.h0 = self.session.solve(self.g0, max_actions=100)
        self.h1 = self.session.solve(self.g1, max_actions=100)

    def test_rejected_sample_is_not_zero_padded_into_denominator(self):
        bad = self.session.backward(
            self.h0, MATERIALS["upstream"][0], max_actions=1, inverse_risk_limit=0.1
        )
        good = self.session.backward(
            self.h1, MATERIALS["upstream"][1], max_actions=100, inverse_risk_limit=100
        )
        window = self.session.begin_window()
        self.assertFalse(window.add(bad))
        self.assertTrue(window.add(good))
        self.assertEqual(window.effective_weight, 2.0)
        aggregate = window.aggregate()
        self.assertIsNotNone(aggregate)
        # Weight 2 / denominator 2 means exactly the one qualified gradient.
        np.testing.assert_allclose(aggregate.W, good.gradients.W, rtol=1e-12)

    def test_two_qualified_samples_use_nonnegative_weight_denominator(self):
        b0 = self.session.backward(
            self.h0, MATERIALS["upstream"][0], max_actions=100, inverse_risk_limit=100
        )
        b1 = self.session.backward(
            self.h1, MATERIALS["upstream"][1], max_actions=100, inverse_risk_limit=100
        )
        self.assertTrue(b0.eligible and b1.eligible)
        window = self.session.begin_window()
        window.add(b0)
        window.add(b1)
        self.assertEqual(window.effective_weight, 3.0)
        expected = (b0.gradients * 1.0 + b1.gradients * 2.0) / 3.0
        np.testing.assert_allclose(window.aggregate().W, expected.W)

    def test_high_gain_rejected_then_gentle_accepted_keeps_window(self):
        bad = self.session.backward(
            self.h0, MATERIALS["upstream"][0], max_actions=1, inverse_risk_limit=0.1
        )
        good = self.session.backward(
            self.h1, MATERIALS["upstream"][1], max_actions=100, inverse_risk_limit=100
        )
        window = self.session.begin_window()
        window.add(bad)
        window.add(good)

        high = next(c for c in MATERIALS["candidate_parameters"] if c["id"] == "high-gain")
        gentle = next(c for c in MATERIALS["candidate_parameters"] if c["id"] == "gentle")

        rejected = self.session.propose_update(high, window)
        self.assertFalse(rejected.accepted)
        self.assertEqual(self.session.generation, 0)
        self.assertAlmostEqual(self.session.params.W[0, 0], 0.22)
        self.assertEqual(window.effective_weight, 2.0)
        self.assertFalse(window.committed)

        accepted = self.session.propose_update(gentle, window)
        self.assertTrue(accepted.accepted)
        self.assertEqual(self.session.generation, 1)
        self.assertTrue(window.committed)
        self.assertAlmostEqual(self.session.params.W[0, 0], 0.3)

    def test_missing_family_is_evidence_gap_not_clear(self):
        session = EquilibriumSession(material_params())
        h = session.solve(self.g1, max_actions=100)
        grad = session.backward(h, MATERIALS["upstream"][1], max_actions=100, inverse_risk_limit=100)
        window = session.begin_window()
        window.add(grad)
        gentle = next(c for c in MATERIALS["candidate_parameters"] if c["id"] == "gentle")
        result = session.propose_update(gentle, window)
        self.assertFalse(result.accepted)
        self.assertEqual(result.status, "stability_evidence_insufficient")
        self.assertEqual(session.generation, 0)

    def test_generation_mismatch_blocks_old_gradient(self):
        good = self.session.backward(
            self.h1, MATERIALS["upstream"][1], max_actions=100, inverse_risk_limit=100
        )
        window = self.session.begin_window()
        window.add(good)
        gentle = next(c for c in MATERIALS["candidate_parameters"] if c["id"] == "gentle")
        self.assertTrue(self.session.propose_update(gentle, window).accepted)

        old_window = LearningWindow(parameter_generation=0)
        self.assertTrue(old_window.add(good))
        second = self.session.propose_update(gentle, old_window)
        self.assertFalse(second.accepted)
        self.assertEqual(second.status, "generation_mismatch")
        self.assertEqual(self.session.generation, 1)

    def test_graph_revisions_cannot_be_mixed_in_one_window(self):
        old = self.session.backward(
            self.h0, MATERIALS["upstream"][0], max_actions=100, inverse_risk_limit=100
        )
        revised_graph = Graph.from_mapping(MATERIALS["graph_revision"])
        h2 = self.session.solve(revised_graph, max_actions=100)
        revised_upstream = np.asarray(MATERIALS["upstream"][0])
        new = self.session.backward(
            h2, revised_upstream, max_actions=100, inverse_risk_limit=100
        )
        self.assertTrue(old.eligible and new.eligible)
        self.assertNotEqual(old.graph_fingerprint, new.graph_fingerprint)

        window = self.session.begin_window()
        first = window.add(old)
        second = window.add(new)
        self.assertTrue(first)
        self.assertFalse(second)
        self.assertEqual(second.reason, "mixed_graph_revisions")
        self.assertEqual(window.effective_weight, 1.0)

    def test_closed_forward_cannot_supply_adjoint(self):
        h = self.session.solve(self.g1, max_actions=100)
        self.session.close_forward(h)
        with self.assertRaises(ClosedForwardError):
            self.session.backward(h, MATERIALS["upstream"][1], max_actions=100, inverse_risk_limit=100)
        rejected = self.session.backward(
            h, MATERIALS["upstream"][1], max_actions=100, inverse_risk_limit=100,
            raise_if_closed=False,
        )
        self.assertFalse(rejected.eligible)
        self.assertEqual(rejected.status, "forward_closed")
        with self.assertRaises(ClosedForwardError):
            h.backward(MATERIALS["upstream"][1])


class BatchOrchestrationTests(unittest.TestCase):
    def test_independent_sample_failures_share_one_window(self):
        session = EquilibriumSession(material_params(), [material_family()])
        window = session.begin_window()
        samples = [
            {
                "graph": MATERIALS["graphs"][0],
                "upstream": MATERIALS["upstream"][0],
                "backward_options": {"max_actions": 1, "inverse_risk_limit": 0.1},
            },
            {
                "graph": MATERIALS["graphs"][1],
                "upstream": MATERIALS["upstream"][1],
                "backward_options": {"max_actions": 100, "inverse_risk_limit": 100},
            },
        ]
        report = process_batch(
            session,
            samples,
            window=window,
            forward_options={"max_actions": 100},
        )
        self.assertEqual(report.viewable_sample_ids, ("g0", "g1"))
        self.assertEqual(report.gradient_rejected_ids, ("g0",))
        self.assertEqual(report.trainable_sample_ids, ("g1",))
        self.assertEqual(window.effective_weight, 2.0)
        payload = report.public_dict()
        self.assertIn("decisions", payload)


class StabilityCertificateTests(unittest.TestCase):
    def test_certified_violated_and_unknown_are_distinct(self):
        family = material_family()
        gentle = Parameters.from_mapping(
            next(c for c in MATERIALS["candidate_parameters"] if c["id"] == "gentle")
        )
        high = Parameters.from_mapping(
            next(c for c in MATERIALS["candidate_parameters"] if c["id"] == "high-gain")
        )
        self.assertEqual(certify_stability(gentle, family).status, "certified")
        self.assertEqual(certify_stability(high, family).status, "violated")
        self.assertEqual(certify_stability(gentle, None).status, "insufficient_evidence")

        weak_family = GraphFamily(
            family_id="weak", max_abs_row_sum=2.0, rho=0.92,
            weight_min=-0.1, weight_max=0.1, max_edges_per_node=1,
        )
        ambiguous = Parameters(W=np.eye(3) * 0.5, U=np.zeros((2, 3)), b=np.zeros(3))
        cert = certify_stability(ambiguous, weak_family)
        self.assertEqual(cert.status, "insufficient_evidence")


class SparseCapacityTests(unittest.TestCase):
    def test_large_sparse_graph_solves_and_backprops_without_dense_paths(self):
        rng = np.random.default_rng(261126)
        n, deg, d, input_dim = 256, 8, 16, 4
        u = np.repeat(np.arange(n), deg)
        v = (u + np.tile(np.arange(1, deg + 1), n)) % n
        w = np.full(n * deg, 1.0 / (deg * 2))
        graph = Graph.from_arrays(
            [f"n-{i}" for i in range(n)],
            rng.normal(0.0, 0.1, (n, input_dim)),
            edge_u=u,
            edge_v=v,
            edge_weight=w,
        )
        params = Parameters(
            W=np.eye(d) * 0.2,
            U=rng.normal(0.0, 0.1, (input_dim, d)),
            b=np.zeros(d),
        )
        fwd = solve_equilibrium(graph, params, max_actions=200)
        self.assertTrue(fwd.accepted, fwd.reason)
        self.assertLessEqual(fwd.residual, fwd.residual_bound)

        grad = implicit_gradients(
            fwd, rng.normal(size=(n, d)), max_actions=200, inverse_risk_limit=1000
        )
        self.assertTrue(grad.eligible, grad.reasons)
        self.assertLessEqual(grad.relative_residual, 1e-6)
        self.assertEqual(grad.risk.method, "structured_sparse_norm_bound")

        import scipy.sparse as sp

        self.assertIsInstance(graph.A, sp.spmatrix)
        with self.assertRaises(Exception):
            graph.dense_A()

    def test_maximum_profile_retains_under_16_mib(self):
        rng = np.random.default_rng(42)
        n, deg, d = 2048, 8, 16
        u = np.repeat(np.arange(n), deg)
        v = (u + np.tile(np.arange(1, deg + 1), n)) % n
        graph = Graph.from_arrays(
            [f"n-{i}" for i in range(n)],
            rng.normal(0.0, 0.1, (n, 4)),
            edge_u=u,
            edge_v=v,
            edge_weight=np.full(n * deg, 1.0 / (deg * 2)),
        )
        params = Parameters(W=np.eye(d) * 0.2, U=rng.normal(0, 0.1, (4, d)), b=np.zeros(d))
        fwd = solve_equilibrium(graph, params, max_actions=200)
        self.assertTrue(fwd.accepted)
        self.assertLess(fwd.retained_bytes, 16 * 1024**2)


class SupportBoundaryTests(unittest.TestCase):
    def test_zero_action_budget_is_rejected(self):
        g = Graph.from_mapping(MATERIALS["graphs"][1])
        fwd = solve_equilibrium(g, material_params(), max_actions=0)
        self.assertFalse(fwd.accepted)
        self.assertEqual(fwd.reason, "action_budget_exhausted")
        self.assertIsNone(fwd.state)

    def test_out_of_support_graph_is_not_solved(self):
        n = 2049
        g = Graph.from_arrays(
            [str(i) for i in range(n)],
            np.zeros((n, 1)),
            edges=[],
        )
        p = Parameters(W=np.eye(8), U=np.zeros((1, 8)), b=np.zeros(8))
        result = solve_equilibrium(g, p, max_actions=10)
        self.assertEqual(result.status, "unsupported")
        self.assertFalse(result.accepted)

    def test_empty_graph_is_accepted_but_excluded_from_training(self):
        g = Graph.from_arrays((), np.zeros((0, 2)), edges=[])
        p = material_params()
        fwd = solve_equilibrium(g, p, max_actions=10)
        self.assertTrue(fwd.accepted)
        grad = implicit_gradients(fwd, None, max_actions=10, inverse_risk_limit=100)
        self.assertFalse(grad.eligible)
        self.assertEqual(grad.status, "empty_excluded")
        self.assertEqual(grad.effective_weight, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
