"""Qualified equilibrium layer for sparse tactile contact graphs.

The package implements the fixed-point solve, matrix-free implicit VJP,
sample/window qualification and family stability evidence.  It deliberately
has no equilibrium-network or automatic-differentiation dependency.
"""

from .errors import (
    EquilibriumError,
    ValidationError,
    UnsupportedGraphError,
    ClosedForwardError,
    BudgetExhausted,
)
from .gradients import ParameterGradients
from .graph import Graph, WarmStart, align_warm_start
from .parameters import Parameters
from .family import GraphFamily, StabilityCertificate, certify_stability, check_graph_membership, lipschitz_upper_bound
from .forward import ForwardReceipt, ForwardResult, solve_equilibrium
from .backward import GradientReceipt, GradientResult, BackwardResult, implicit_gradients, jacobian_norm_bounds
from .window import LearningWindow, UpdateResult, UpdateDecision, aggregate_gradients
from .session import EquilibriumSession, ForwardHandle
from .batch import BatchReport, SampleDecision, process_batch

# Descriptive aliases.
EquilibriumLayer = EquilibriumSession
QualifiedEquilibriumLayer = EquilibriumSession
ContactGraph = Graph
SparseGraph = Graph
EquilibriumParameters = Parameters
equilibrium_forward = solve_equilibrium
equilibrium_backward = implicit_gradients
compute_implicit_gradients = implicit_gradients
check_stability = certify_stability

__all__ = [
    "EquilibriumError",
    "ValidationError",
    "UnsupportedGraphError",
    "ClosedForwardError",
    "BudgetExhausted",
    "ParameterGradients",
    "Graph",
    "WarmStart",
    "align_warm_start",
    "Parameters",
    "GraphFamily",
    "StabilityCertificate",
    "certify_stability",
    "check_graph_membership",
    "lipschitz_upper_bound",
    "ForwardReceipt",
    "ForwardResult",
    "solve_equilibrium",
    "GradientReceipt",
    "GradientResult",
    "BackwardResult",
    "implicit_gradients",
    "jacobian_norm_bounds",
    "LearningWindow",
    "aggregate_gradients",
    "UpdateResult",
    "UpdateDecision",
    "EquilibriumSession",
    "EquilibriumLayer",
    "QualifiedEquilibriumLayer",
    "ContactGraph",
    "SparseGraph",
    "EquilibriumParameters",
    "ForwardHandle",
    "BatchReport",
    "SampleDecision",
    "process_batch",
    "equilibrium_forward",
    "equilibrium_backward",
    "compute_implicit_gradients",
    "check_stability",
]

__version__ = "1.0.0"
