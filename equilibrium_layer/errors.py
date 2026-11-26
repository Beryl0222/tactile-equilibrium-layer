"""Exception types used by the equilibrium layer."""


class EquilibriumError(Exception):
    """Base class for programmer/configuration errors."""


class ValidationError(EquilibriumError, ValueError):
    """An input graph, parameter or upstream specification is invalid."""


class UnsupportedGraphError(EquilibriumError):
    """The request is outside the declared CPU support envelope."""


class ClosedForwardError(EquilibriumError):
    """An adjoint was requested for a forward explicitly closed by the caller."""


class BudgetExhausted(EquilibriumError):
    """Raised by optional low-level helpers when an action budget is exhausted.

    High-level solve/gradient calls normally return a rejected receipt instead
    of raising this exception.
    """
