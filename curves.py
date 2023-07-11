from typing import Tuple
import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
from math import pi


class Curve:
    def __init__(self, t: sym.Symbol, x: sym.Expr, y: sym.Expr, t_span: Tuple[float, float]) -> None:
        self.t = t
        self.x = x
        self.y = y
        self.t_span = t_span
        
        self.x_f = sym.lambdify(t, x)
        self.y_f = sym.lambdify(t, y)

def plot_curve(curve: Curve, resolution: int=100):
    figname = 'curve'
    plt.close(figname)

    fig, ax = plt.subplots(num=figname)
    ax.set_aspect('equal')
    fig.canvas.header_visible = False

    t = np.linspace(curve.t_span[0], curve.t_span[1], resolution)
    x = curve.x_f(t)
    y = curve.y_f(t)
    ax.plot(x, y)


def circle() -> Curve:
    t = sym.Symbol('t', real=True)
    x = sym.cos(t)
    y = sym.sin(t)
    t_span = [-1.45*pi, 0.45*pi]
    curve = Curve(t, x, y, t_span)
    return curve

def tear() -> Curve:
    t = sym.Symbol('t', real=True)
    x = 0.3*sym.cos(3*t)
    y = 2*sym.sin(t)
    t_span = [-2.8, -pi+2.8]
    curve = Curve(t, x, y, t_span)
    return curve

def hill() -> Curve:
    t = sym.Symbol('t', real=True)
    x = t
    y = 2*t**4 - 1.9*t**2 + 0.5
    t_span = [-1, 1]
    curve = Curve(t, x, y, t_span)
    return curve


if __name__ == '__main__':
    curve = hill()
    plot_curve(curve)
    plt.show()