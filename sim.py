from typing import Tuple, Dict
import sympy as sym
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import ipywidgets as widgets


class Curve:
    def __init__(self, t: sym.Symbol, x: sym.Expr, y: sym.Expr, t_span: Tuple[float, float]) -> None:
        self.t = t
        self.x = x
        self.y = y
        self.t_span = t_span
        
        self.x_f = sym.lambdify(t, x)
        self.y_f = sym.lambdify(t, y)


class CurveModeler:
    def __init__(self, t: sym.Symbol, f_params: Dict[str, sym.Symbol], x: sym.Expr, y: sym.Expr, slider_params: Dict[str, Dict[str, float]]) -> None:
        self._check_matching_keys(f_params.keys(), slider_params.keys())
        self.t = t
        self.f_params = f_params
        self.x = x
        self.y = y
        self.slider_params = slider_params
        
        self.set_numeric_f_params({key: slider['value'] for (key, slider) in self.slider_params.items()},
                                  (self.slider_params['t_0']['value'], self.slider_params['t_n']['value']))
        
        self.x_f = sym.lambdify([self.t, self.f_params.values()], self.x)
        self.y_f = sym.lambdify([self.t, self.f_params.values()], self.y)
    
    @staticmethod
    def _check_matching_keys(f_params_keys, slider_params_keys):
        f_params_keys = tuple(f_params_keys)
        slider_params_keys = tuple(key for key in slider_params_keys if key not in ('t_0', 't_n'))
        if f_params_keys != slider_params_keys:
            raise ValueError("f_params and slider_params dict keys don't match or their order doesn't match")
    
    def set_numeric_f_params(self, f_params_num: Dict[str, float], t_span: Tuple[float, float]):
        self.f_params_num = f_params_num
        self.t_span = t_span

    @staticmethod
    def _make_sliders(params: Dict[str, Dict[str, float]]) -> Dict[str, widgets.FloatSlider]:
        sliders = {}
        for item in params.items():
            sliders[item[0]] = widgets.FloatSlider(**item[1])
        return sliders

    def model_curve(self):
        fig, ax = plt.subplots()
        fig.set_size_inches(6, 6)
        ax.set_aspect('equal', adjustable='datalim')
        fig.canvas.header_visible = False

        lines = plt.plot(0, 0)
        plt.ion()

        def plot(**params):
            # split params into groups
            # function params, that is all but t_0, t_n
            f_params = {key: params[key] for key in params.keys() if key not in ('t_0', 't_n')}
            # t_0, t_n
            t_span = {key: params[key] for key in params.keys() if key in ('t_0', 't_n')}

            t = np.linspace(t_span['t_0'], t_span['t_n'], int((t_span['t_n']-t_span['t_0'])*100+1))

            # this is order-sensitive
            x = self.x_f(t, f_params.values())
            y = self.y_f(t, f_params.values())
            self.set_numeric_f_params(f_params, tuple(t_span.values()))

            lines[0].set_data(x, y)
            ax.relim()
            ax.autoscale_view()
        
        # this ensures t_n isn't smaller than t_0
        def update_t_n_range(*args):
            sliders['t_n'].min = sliders['t_0'].value + sliders['t_n'].step     
        
        sliders = self._make_sliders(self.slider_params)
        sliders['t_0'].observe(update_t_n_range, 'value')
        widgets.interact(plot, **sliders)
    
    def generate_curve(self) -> Curve:
        x = self.x
        y = self.y
        # requires self.f_params and self.f_params_num to have matching values in the same keys
        for key in self.f_params.keys():
            x = x.subs(self.f_params[key], self.f_params_num[key])
            y = y.subs(self.f_params[key], self.f_params_num[key])
        curve = Curve(self.t, x, y, self.t_span)
        return curve
           

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

    def _dyn_eq_jac_f(self, t, state, g):
        p, v = state
        return [[0, 1], [g*self._a_f.derivative()(p), 0]]

    def simulate(self, g=9.81, t_max=30, t_res=100, method='RK45', rtol=1e-4, atol=1e-6, print_solver_output=False):
        '''
        g - gravity constant, default: 9.81 [m/s**2]

        t_max - max simulation time, default: 30 [s]

        t_res - time resolution, default: 100 [Hz]

        method - integration method to pass to solver,
        default: 'RK45';
        see https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html

        rtol - relative error tolerance, default: 1e-4

        atol - absolute error tolerance, defualt: 1e-6

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
        # some solvers use jacobian
        if method in ['Radau', 'BDF', 'LSODA']:
            sol = solve_ivp(self._dyn_eq_f, (0, t_max), initial_conditions, method=method, args=(g,), t_eval=t, 
                        jac=self._dyn_eq_jac_f, rtol=rtol, atol=atol)
        else:
            sol = solve_ivp(self._dyn_eq_f, (0, t_max), initial_conditions, method=method, args=(g,), t_eval=t, 
                        rtol=rtol, atol=atol)
        if print_solver_output:
            print(sol)
        t = sol.t
        # this gives p and p'
        p = sol.y
        # this gives x''
        a = self._dyn_eq_f(t, p, g)[1]
        # concatenate a into p for array (p, p', p'')
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

def calc_energy(p, y, g=9.81):
    kinetic_energy = (p[1,:]**2/2)
    potential_energy = (y - y.min())*g
    total_energy = kinetic_energy + potential_energy
    return total_energy, kinetic_energy, potential_energy

# may not work in real time if time resolution is too large
def plot_sim_results(t, p, path_xy, x, y, animated=True, speed=1.0):
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
    
    e_total, e_kin, e_pot = calc_energy(p, y)

    v = p[1]
    a = p[2]
    p = p[0]

    if animated:
        t_plt = []
        x_plt = []
        y_plt = []
        p_plt = []
        v_plt = []
        a_plt = []
        e_total_plt = []
        e_kin_plt = []
        e_pot_plt = []
    else:
        t_plt = t
        x_plt = x
        y_plt = y
        p_plt = p
        v_plt = v
        a_plt = a
        e_total_plt = e_total
        e_kin_plt = e_kin
        e_pot_plt = e_pot

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
    e_lim = expand_limits((min(e_total.min(), e_kin.min(), e_pot.min()), max(e_total.max(), e_kin.max(), e_pot.max())), 0.05)

    # y(x)
    x_lim, y_lim = equalize_axis_scales(x_lim, y_lim)
    axs[0][0].set_xlabel('x')
    axs[0][0].set_ylabel('y')
    axs[0][0].set_xlim(x_lim)
    axs[0][0].set_ylim(y_lim)
    axs[0][0].axis("scaled")
    line_yx, = axs[0][0].plot(path_xy[0], path_xy[1])
    if animated:
        point_yx, = axs[0][0].plot(x_plt, y_plt, 'o')

    # e(t)
    axs[0][1].set_xlabel('t')
    axs[0][1].set_ylabel('E')
    axs[0][1].set_xlim(t_lim)
    axs[0][1].set_ylim(e_lim)
    line_e_total_t, = axs[0][1].plot(t_plt, e_total_plt)
    line_e_kin_t, = axs[0][1].plot(t_plt, e_kin_plt)
    line_e_pot_t, = axs[0][1].plot(t_plt, e_pot_plt)
    axs[0][1].legend(['Total Energy', 'Kinetic Energy', 'Potential Energy'])

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


    def init():
        point_yx.set_data([], [])
        line_e_total_t.set_data([], [])
        line_e_kin_t.set_data([], [])
        line_e_pot_t.set_data([], [])
        line_pt.set_data([], [])
        line_vt.set_data([], [])
        line_at.set_data([], [])
        return point_yx, line_e_total_t, line_e_kin_t, line_e_pot_t, line_pt, line_vt, line_at

    def animate(i):
        point_yx.set_data(x[i], y[i])
        
        i += 1
        line_e_total_t.set_data(t[:i], e_total[:i])
        line_e_kin_t.set_data(t[:i], e_kin[:i])
        line_e_pot_t.set_data(t[:i], e_pot[:i])

        line_pt.set_data(t[:i], p[:i])
        line_vt.set_data(t[:i], v[:i])
        line_at.set_data(t[:i], a[:i])

        line_pt.set_xdata(t_plt)
        line_pt.set_ydata(p_plt)

        return point_yx, line_e_total_t, line_e_kin_t, line_e_pot_t, line_pt, line_vt, line_at

    if animated:
        return FuncAnimation(plt.gcf(), animate, init_func=init, frames=len(x), interval=int((t[1]-t[0])*1000/speed), 
                             repeat=False, blit=True)

        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    if animated:
        plt.ion()
        return FuncAnimation(plt.gcf(), animate, frames=len(x), interval=int((t[1]-t[0])*1000/speed), repeat=False)


if __name__ == '__main__':
    sim = Sim(Curve())
    t, p, path_xy, x, y = sim.simulate(t_max=10.0)
    plot_sim_results(t, p, path_xy, x, y, animated=False)
    plt.show()