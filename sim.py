from typing import Tuple, Dict
import sympy as sym
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import ipywidgets as widgets
from time import time
from math import ceil


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
        # close previous instance of this figure, if it exists
        # the name should be unique
        figname = 'model_curve'
        plt.close(figname)

        fig, ax = plt.subplots(num=figname)
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

        print_solver_output - self-explaining, I guess

        ---

        returns:

        t - evaluation times array

        p - (p, p', p'') - position, speed and acceleration arrays tuple in path coordinates, ordered by time

        curve_xy - (x, y) - curve on which the body moves; tuple of arrays

        x - x position array, ordered by time

        y - y position array, ordered by time
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
        curve_xy = self.eval_path_points()
        return t, p, curve_xy, x, y

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


# this is to evaluate animation execution
anim_frame_time_list = []

def plot_sim_results(t: np.ndarray, p: np.ndarray, curve_xy: np.ndarray, x: np.ndarray, y: np.ndarray, 
                     optional_features=('e_total', 'e_kin', 'e_pot', 'p', 'v', 'a'), 
                     animated=True, interval=100, speed=1.0, eval_animation=False) -> None | FuncAnimation:
    '''
    t - time array

    p - (p, p', p'') - position, speed and acceleration arrays tuple in path coordinates, ordered by time

    curve_xy - (x, y) - curve on which the body moves; tuple of arrays

    x - x position array, ordered by time

    y - y position array, ordered by time

    optional_features - additional features to plot; all supported features in default argument; tuple of strings

    animated - True for animated plot (only for Jupyter notebooks), False for static

    interval - amount of time between animation frames [ms]

    speed - animation speed multiplier

    eval_animation - whether to evaluate animation rendering time (to check if it isn't lagging behind simulation data)

    ---

    Returns matplotlib.animation.FuncAnimation for animated=True, none otherwise
    '''

    # close previous instance of this figure, if it exists
    # the name should be unique
    figname = 'sim_results'
    plt.close(figname)

    anim_frame_time_list.clear()

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

    fig, (ax_yx, ax_pt, ax_et) = plt.subplots(1,3, num=figname)
    fig.set_size_inches(14,4)
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
    ax_yx.set_xlabel('x')
    ax_yx.set_ylabel('y')
    ax_yx.set_xlim(x_lim)
    ax_yx.set_ylim(y_lim)
    ax_yx.axis("scaled")
    line_yx, = ax_yx.plot(curve_xy[0], curve_xy[1])
    if animated:
        point_yx, = ax_yx.plot(x_plt, y_plt, 'o')

    # p(t)
    ax_pt.set_xlabel('t')
    ax_pt.set_ylabel('p')
    ax_pt.set_xlim(t_lim)
    ax_pt.set_ylim(p_lim)
    line_pt, = ax_pt.plot(t_plt, p_plt, color='b', label='p')
    ax_pt.yaxis.label.set_color(line_pt.get_color())

    # v(t)
    ax_vt = ax_pt.twinx()
    ax_vt.set_ylabel('v')
    ax_vt.set_ylim(v_lim)
    line_vt, = ax_vt.plot(t_plt, v_plt, color='orange', label='v')
    ax_vt.spines['left'].set_position(('outward', 40))
    ax_vt.spines['left'].set_visible(True)
    ax_vt.yaxis.set_label_position('left')
    ax_vt.yaxis.set_ticks_position('left')
    ax_vt.yaxis.label.set_color(line_vt.get_color())

    # a(t)
    ax_at = ax_pt.twinx()
    ax_at.set_ylabel('a')
    ax_at.set_ylim(a_lim)
    line_at, = ax_at.plot(t_plt, a_plt, color='green', label='a')
    ax_at.spines['right']
    ax_at.yaxis.label.set_color(line_at.get_color())

    ax_pt.legend(handles=[line_pt, line_vt, line_at])
    
    # e(t)
    ax_et.set_xlabel('t')
    ax_et.set_ylabel('E')
    ax_et.set_xlim(t_lim)
    ax_et.set_ylim(e_lim)
    line_e_total_t, = ax_et.plot(t_plt, e_total_plt)
    line_e_kin_t, = ax_et.plot(t_plt, e_kin_plt)
    line_e_pot_t, = ax_et.plot(t_plt, e_pot_plt)
    ax_et.legend(['Total Energy', 'Kinetic Energy', 'Potential Energy'])

    plt.tight_layout()

    # what to plot for each feature key (plot, (data_x, data_y))
    supported_features = {'e_total': (line_e_total_t, (t, e_total)),
                          'e_kin': (line_e_kin_t, (t, e_kin)),
                          'e_pot': (line_e_pot_t, (t, e_pot)),
                          'p': (line_pt, (t, p)),
                          'v': (line_vt, (t, v)),
                          'a': (line_at, (t, a))}
    
    features_data = []
    for feature in optional_features:
        if feature in supported_features.keys():
            features_data.append(supported_features[feature])
    features_data = tuple(features_data)

    def init():
        point_yx.set_data([], [])

        for plot, data in features_data:
            plot.set_data([], [])
        
        return point_yx, *features_data

    def animate(i):
        point_yx.set_data(x[i], y[i])
        
        i += 1
        for plot, data in features_data:
            plot.set_data(data[0][:i], data[1][:i])

        # there should be "if eval_animation:" here, but it's too slow
        # TODO: or is it?
        anim_frame_time_list.append(time())

        return point_yx, *features_data

    if animated:
        data_interval = int((t[1]-t[0])*1000/speed)
        # data denser than animation rate
        if data_interval < interval:
            print("Data is denser than animation rate. Some frames will be skipped.")
            if interval/data_interval % 1 != 0:
                print("Data rate is not integer multiply of animation rate. Animation rate will decrease accordingly.")
            frame_multiplier = ceil(interval/data_interval)
            interval = int(data_interval*frame_multiplier)
            frames = [i*frame_multiplier for i in range(ceil(len(t)/frame_multiplier))]
        # data sparser than animation rate
        elif data_interval > interval:
            print("Data is sparser than animation rate. Animiation rate will be lower.")
            interval = int(data_interval)
            frames = len(t)
        # data rate same as animation rate (everything fine)
        else:
            frames = len(t)
    
        if eval_animation:
            print(f"expected interval [ms]: {interval}")
            print(f"expected total time [s]: {(t[-1]-t[0])/speed}")
        
        return FuncAnimation(plt.gcf(), animate, init_func=init, frames=frames, interval=interval, 
                             repeat=False, blit=True)

def eval_frame_processing_time():
    if not anim_frame_time_list:
        print("No time data to evaluate.")
        return

    anim_frame_time_arr = np.array(anim_frame_time_list)
    intervals = anim_frame_time_arr[1:] - anim_frame_time_arr[:-1]
    total_time = anim_frame_time_arr[-1] - anim_frame_time_arr[0]
    print(f"average interval [ms]: {np.average(intervals)*1000}")
    print(f"total time [s]: {total_time}")


if __name__ == '__main__':
    t = sym.Symbol('t', real=True)
    x = 0.3*sym.cos(3*t)
    y = sym.sin(t)
    t_span = [-2.8, -np.pi+2.8]
    curve = Curve(t, x, y, t_span)
    sim = Sim(curve)
    t, p, path_xy, x, y = sim.simulate(t_max=10.0)
    plot_sim_results(t, p, path_xy, x, y, animated=False)
    plt.show()