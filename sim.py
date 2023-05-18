import sympy as sym
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Curve:
    def __init__(self) -> None:
        self.t = sym.Symbol('t', real=True)

        self.x = 0.3*sym.cos(3*self.t)
        self.y = 2*sym.sin(1*self.t)
        self.t_span = [-2.8, -np.pi+2.8]
        
        self.x_f = sym.lambdify(self.t, self.x)
        self.y_f = sym.lambdify(self.t, self.y)


class Sim:
    def __init__(self, curve: Curve) -> None:
        self._curve = curve

        p_integrand = sym.sqrt(sym.diff(self._curve.x, self._curve.t)**2 + sym.diff(self._curve.y, self._curve.t)**2)
        p_integrand_f = sym.lambdify(self._curve.t, p_integrand)

        # curve steepness
        k = (sym.diff(self._curve.y, self._curve.t) / sym.Abs(sym.diff(self._curve.x, self._curve.t))).simplify()
        
        # this is acceleration from steepness in g units
        # it handles the vertical line situation (infinite k)
        a = sym.Piecewise(
            (-sym.sign(k), sym.Eq(sym.Abs(k), sym.oo)),
            (-k/(1+k**2)**0.5, True)).simplify()
        a_f = sym.lambdify(self._curve.t, a)
        
        # use these tables to interpolate continuous a(p) and t(p) functions
        # TODO: try to make these uniformly distributed in p, not t
        t_span = self._curve.t_span
        t = np.linspace(t_span[0], t_span[1], int((t_span[1]-t_span[0])*100+1))
        # TODO: maybe doing it in steps wold be computationally better than starting from t0 every time, need to check
        p_table = np.array([quad(p_integrand_f, t[0], u_lim)[0] for u_lim in t])
        a_table = a_f(t)

        # a(p)
        self._a_f = CubicSpline(p_table, a_table)
        # t(p)
        self._t_p_f = CubicSpline(p_table, t)

    def _dyn_eq_f(self, t, state, g):
            p, v = state
            return [v, g*self._a_f(p)]

    def simulate(self, g=9.81, t_max=30, t_res=100):
        '''
        g - gravity constant, default: 9.81 [m/s**2]

        t_max - max simulation time, default: 30 [s]

        t_res - time resolution, default: 10 [Hz]

        returns:

        t - evaluation times array

        (p, p', p'') - states array in path coordinates

        path_xy - path points

        x - x positions array

        y - y positions array
        '''
        # evaluation times vector
        t = np.linspace(0, t_max, int(t_max*t_res))
        initial_conditions = (0, 0)  # (p(0) [m], p'(0) [m/s])
        # solve dynamics equation numerically with given parameters
        sol = solve_ivp(self._dyn_eq_f, (0, t_max), initial_conditions, method='Radau', args=(g,), t_eval=t)
        # print(sol)
        t = sol.t
        # this gives p and p'
        p = sol.y
        # this gives x''
        a = self._dyn_eq_f(t, p, g)[1]
        # concatenate a into x for array (x, x', x'')
        p = np.row_stack((p,a))
        x, y = self.evaluate_x_y(p[0])
        path_xy = self.eval_path_points()
        return t, p, path_xy, x, y

    def evaluate_x_y(self, p_arr):
        t_arr = self._t_p_f(p_arr)
        x = self._curve.x_f(t_arr)
        y = self._curve.y_f(t_arr)
        return x, y

    def eval_path_points(self):
        t_span = self._curve.t_span
        t = np.linspace(t_span[0], t_span[1], int((t_span[1]-t_span[0])*100+1))
        x = self._curve.x_f(t)
        y = self._curve.y_f(t)
        return x, y

# may not work in real time if time resolution is too large
def plot_sim_results(t, p, path_xy, x, y):
    # make axis limits a little bigger than necessary
    def expand_limits(lim, amount):
        additional_range = (lim[1] - lim[0]) * amount
        if additional_range == 0:
            additional_range = 1 * amount
        return (lim[0] - additional_range, lim[1] + additional_range)
    
    # fix axis limits so scale is the same on both axes
    def equalize_axis_scales(x_lim, y_lim):
        x_range = x_lim[1] - x_lim[0]
        y_range = y_lim[1] - y_lim[0]
        if x_range > y_range:
            y_middle = (y_lim[0] + y_lim[1]) / 2
            y_lim = (y_middle - x_range/2, y_middle + x_range/2)
        elif y_range > x_range:
            x_middle = (x_lim[0] + x_lim[1]) / 2
            x_lim = (x_middle - y_range/2, x_middle + y_range/2)
        return x_lim, y_lim
    

    v = p[1]
    a = p[2]
    p = p[0]

    t_plt = []
    x_plt = []
    y_plt = []
    p_plt = []
    v_plt = []
    a_plt = []

    fig, axs = plt.subplots(2,3)
    fig.set_size_inches(10,7)
    fig.canvas.header_visible = False

    # make static plot limits
    t_lim = expand_limits((t.min(), t.max()), 0.05)
    x_lim = expand_limits((x.min(), x.max()), 0.05)
    y_lim = expand_limits((y.min(), y.max()), 0.05)
    p_lim = expand_limits((p.min(), p.max()), 0.05)
    v_lim = expand_limits((v.min(), v.max()), 0.05)
    a_lim = expand_limits((a.min(), a.max()), 0.05)

    # x(t)
    axs[0][0].set_xlabel('t')
    axs[0][0].set_ylabel('x')
    axs[0][0].set_xlim(t_lim)
    axs[0][0].set_ylim(x_lim)
    line_xt, = axs[0][0].plot(t_plt, x_plt)

    # y(t)
    axs[0][1].set_xlabel('t')
    axs[0][1].set_ylabel('y')
    axs[0][1].set_xlim(t_lim)
    axs[0][1].set_ylim(y_lim)
    line_yt, = axs[0][1].plot(t_plt, y_plt)

    # y(x)
    x_lim, y_lim = equalize_axis_scales(x_lim, y_lim)
    axs[0][2].set_xlabel('x')
    axs[0][2].set_ylabel('y')
    axs[0][2].set_xlim(x_lim)
    axs[0][2].set_ylim(y_lim)
    axs[0][2].axis("scaled")
    line_yx, = axs[0][2].plot(path_xy[0], path_xy[1])
    point_yx, = axs[0][2].plot(x_plt, y_plt, 'o')

    # p(t)
    axs[1][0].set_xlabel('t')
    axs[1][0].set_ylabel('p')
    axs[1][0].set_xlim(t_lim)
    axs[1][0].set_ylim(p_lim)
    line_pt, = axs[1][0].plot(t_plt, p_plt)

    # v(t)
    axs[1][1].set_xlabel('t')
    axs[1][1].set_ylabel('v')
    axs[1][1].set_xlim(t_lim)
    axs[1][1].set_ylim(v_lim)
    line_vt, = axs[1][1].plot(t_plt, v_plt)

    # a(t)
    axs[1][2].set_xlabel('t')
    axs[1][2].set_ylabel('a')
    axs[1][2].set_xlim(t_lim)
    axs[1][2].set_ylim(a_lim)
    line_at, = axs[1][2].plot(t_plt, a_plt)

    plt.tight_layout()
    plt.ion()

    def animate(i):
        t_plt.append(t[i])
        x_plt.append(x[i])
        y_plt.append(y[i])
        p_plt.append(p[i])
        v_plt.append(v[i])
        a_plt.append(a[i])

        line_xt.set_xdata(t_plt)
        line_xt.set_ydata(x_plt)

        line_yt.set_xdata(t_plt)
        line_yt.set_ydata(y_plt)

        point_yx.set_xdata(x[i])
        point_yx.set_ydata(y[i])

        line_pt.set_xdata(t_plt)
        line_pt.set_ydata(p_plt)

        line_vt.set_xdata(t_plt)
        line_vt.set_ydata(v_plt)

        line_at.set_xdata(t_plt)
        line_at.set_ydata(a_plt)

        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    return FuncAnimation(plt.gcf(), animate, frames=len(x), interval=int((t[1]-t[0])*1000), repeat=False)