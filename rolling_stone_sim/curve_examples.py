from curve import *


def teardrop_bezier() -> Curve:
    P = np.array([[-1, -1], [10, -5], [-10, -5], [1, -1]])
    curve = CubicBezierCurve(P)
    return curve


def circle() -> Curve:
    t = sym.Symbol("t", real=True)
    x = sym.cos(t)
    y = sym.sin(t)
    t_span = [-1.45 * pi, 0.45 * pi]
    curve = Curve(t, x, y, t_span)
    return curve


def teardrop() -> Curve:
    t = sym.Symbol("t", real=True)
    x = 0.3 * sym.cos(3 * t)
    y = 2 * sym.sin(t)
    t_span = [-2.8, -pi + 2.8]
    curve = Curve(t, x, y, t_span)
    return curve


def hill() -> Curve:
    t = sym.Symbol("t", real=True)
    x = t
    y = 2 * t**4 - 1.9 * t**2 + 0.5
    t_span = [-1, 1]
    curve = Curve(t, x, y, t_span)
    return curve


if __name__ == "__main__":
    curve = teardrop_bezier()
    plot_curve(curve)
    plt.show()
