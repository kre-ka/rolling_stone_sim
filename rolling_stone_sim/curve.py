from typing import Tuple
import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
from math import pi


class Curve:
    def __init__(
        self, t: sym.Symbol, x: sym.Expr, y: sym.Expr, t_span: Tuple[float, float]
    ) -> None:
        self.t = t
        self.x = x
        self.y = y
        self.t_span = t_span

        self.x_f = sym.lambdify(t, x)
        self.y_f = sym.lambdify(t, y)


class CubicBezierCurve(Curve):
    def __init__(self, P: np.ndarray) -> None:
        """
        P - array of control point coordinates; np.ndarray of shape (4, 2)
        """
        t = sym.Symbol("t", real=True)
        x, y = (
            (1 - t) ** 3 * P[0]
            + 3 * t * (1 - t) ** 2 * P[1]
            + 3 * t**2 * (1 - t) * P[2]
            + t**3 * P[3]
        )
        t_span = (0, 1)
        super().__init__(t, x, y, t_span)


def plot_curve(curve: Curve, resolution: int = 100):
    figname = "curve"
    plt.close(figname)

    fig, ax = plt.subplots(num=figname)
    ax.set_aspect("equal")
    fig.canvas.header_visible = False

    t = np.linspace(curve.t_span[0], curve.t_span[1], resolution)
    x = curve.x_f(t)
    y = curve.y_f(t)
    ax.plot(x, y)
