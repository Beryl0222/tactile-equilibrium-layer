"""Broad compatibility surface for ``from solution import ...`` callers."""

from equilibrium_layer import *  # noqa: F401,F403
from equilibrium_layer import __version__  # noqa: F401
from equilibrium_layer.backward import implicit_gradients
from equilibrium_layer.family import certify_stability
from equilibrium_layer.forward import solve_equilibrium

equilibrium_forward = solve_equilibrium
equilibrium_backward = implicit_gradients
compute_implicit_gradients = implicit_gradients
check_stability = certify_stability
