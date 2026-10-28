"""Safe expression-based two-state ODE solver using classical RK4."""

from __future__ import annotations

import ast
from collections.abc import Callable

import numpy as np


_FUNCTIONS: dict[str, Callable[[float], float]] = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "exp": np.exp,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "log": np.log,
}
_VARIABLES = {"t", "x", "v"}


class _ExpressionValidator(ast.NodeVisitor):
    """Allow only arithmetic expressions, approved functions, and known names."""

    def __init__(self, names: set[str]) -> None:
        self.names = names

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        if node.id not in self.names and node.id not in _FUNCTIONS:
            raise ValueError(f"Unknown symbol: {node.id}")

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
            raise ValueError("Only these functions are allowed: sin, cos, tan, exp, sqrt, abs, log.")
        if len(node.args) != 1 or node.keywords:
            raise ValueError("Functions must have exactly one argument.")
        self.generic_visit(node)

    def generic_visit(self, node: ast.AST) -> None:
        allowed = (
            ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Load,
            ast.Constant, ast.Call, ast.Add, ast.Sub, ast.Mult, ast.Div,
            ast.Pow, ast.Mod, ast.USub, ast.UAdd,
        )
        if not isinstance(node, allowed):
            raise ValueError("Only arithmetic expressions are supported.")
        super().generic_visit(node)


def compile_expression(expression: str, parameter_names: set[str]) -> Callable[[float, float, float, dict[str, float]], float]:
    """Compile a constrained arithmetic expression for a simulation equation."""
    expression = expression.strip()
    if not expression:
        raise ValueError("An equation cannot be empty.")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid equation syntax: {exc.msg}") from exc
    _ExpressionValidator(_VARIABLES | parameter_names).visit(tree)
    code = compile(tree, "<equation>", "eval")

    def evaluate(t: float, x: float, v: float, parameters: dict[str, float]) -> float:
        values = {"t": t, "x": x, "v": v, **parameters, **_FUNCTIONS}
        result = float(eval(code, {"__builtins__": {}}, values))  # noqa: S307 - AST is restricted above.
        if not np.isfinite(result):
            raise ValueError("Equation produced a non-finite value.")
        return result

    return evaluate


def solve_two_state(
    dx_expression: str,
    dv_expression: str,
    parameters: dict[str, float],
    x0: float,
    v0: float,
    duration: float,
    samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve dx/dt and dv/dt with fixed-step fourth-order Runge–Kutta."""
    if duration <= 0 or not np.isfinite(duration):
        raise ValueError("Duration must be a positive finite number.")
    if samples < 2:
        raise ValueError("Samples must be at least 2.")
    if not np.isfinite(x0) or not np.isfinite(v0):
        raise ValueError("Initial values must be finite.")
    if any(not np.isfinite(value) for value in parameters.values()):
        raise ValueError("All parameters must be finite.")

    names = set(parameters)
    f_x = compile_expression(dx_expression, names)
    f_v = compile_expression(dv_expression, names)
    t = np.linspace(0.0, duration, int(samples))
    x = np.empty_like(t)
    v = np.empty_like(t)
    derivative = np.empty_like(t)
    x[0], v[0] = x0, v0

    for i in range(len(t) - 1):
        h = t[i + 1] - t[i]
        try:
            k1x, k1v = f_x(t[i], x[i], v[i], parameters), f_v(t[i], x[i], v[i], parameters)
            k2x = f_x(t[i] + h / 2, x[i] + h * k1x / 2, v[i] + h * k1v / 2, parameters)
            k2v = f_v(t[i] + h / 2, x[i] + h * k1x / 2, v[i] + h * k1v / 2, parameters)
            k3x = f_x(t[i] + h / 2, x[i] + h * k2x / 2, v[i] + h * k2v / 2, parameters)
            k3v = f_v(t[i] + h / 2, x[i] + h * k2x / 2, v[i] + h * k2v / 2, parameters)
            k4x = f_x(t[i] + h, x[i] + h * k3x, v[i] + h * k3v, parameters)
            k4v = f_v(t[i] + h, x[i] + h * k3x, v[i] + h * k3v, parameters)
        except (OverflowError, ValueError) as exc:
            raise ValueError(f"Solver stopped near t = {t[i]:.5g}: {exc}") from exc
        x[i + 1] = x[i] + h * (k1x + 2 * k2x + 2 * k3x + k4x) / 6
        v[i + 1] = v[i] + h * (k1v + 2 * k2v + 2 * k3v + k4v) / 6
        if not np.isfinite(x[i + 1]) or not np.isfinite(v[i + 1]):
            raise ValueError(f"Solver became unstable near t = {t[i + 1]:.5g}. Try a shorter duration or more samples.")

    for i, time in enumerate(t):
        derivative[i] = f_v(time, x[i], v[i], parameters)
    return t, x, v, derivative
