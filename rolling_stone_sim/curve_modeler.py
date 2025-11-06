from typing import Tuple, Dict
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
import ipywidgets as widgets

from curve import Curve


class CurveModeler:
    def __init__(
        self,
        t: sym.Symbol,
        f_params: Dict[str, sym.Symbol],
        x: sym.Expr,
        y: sym.Expr,
        slider_params: Dict[str, Dict[str, float]],
    ) -> None:
        self._check_param_keys(f_params.keys(), slider_params.keys())
        self.t = t
        self.f_params = f_params
        self.x = x
        self.y = y
        # this one requires some more tweaking, hence dedicated method
        self._set_slider_params(slider_params)

        self.set_numeric_f_params(
            {key: slider["value"] for (key, slider) in self.slider_params.items()},
            (self.slider_params["t_0"]["value"], self.slider_params["t_n"]["value"]),
        )

        self.x_f = sym.lambdify([self.t, self.f_params.values()], self.x)
        self.y_f = sym.lambdify([self.t, self.f_params.values()], self.y)

    @staticmethod
    def _check_param_keys(f_params_keys, slider_params_keys):
        if "t" not in slider_params_keys:
            raise ValueError("No 't' key in slider_params")
        f_params_keys = tuple(f_params_keys)
        slider_params_keys = tuple(
            key for key in slider_params_keys if key not in ("t")
        )
        if f_params_keys != slider_params_keys:
            raise ValueError(
                "f_params and slider_params dict keys don't match or their order doesn't match"
            )

    def _set_slider_params(self, slider_params_in: Dict[str, Dict[str, float]]):
        # all function params intact (that is all but 't')
        slider_params_out = {
            key: slider_params_in[key]
            for key in slider_params_in.keys()
            if key not in ("t")
        }
        # split 't' into 't_0' and 't_n'
        t = slider_params_in["t"]
        slider_params_out["t_0"] = {
            "min": t["min"],
            "max": t["max"] - t["step"],
            "step": t["step"],
            "value": t["value"][0],
        }
        slider_params_out["t_n"] = {
            "min": t["min"] + t["step"],
            "max": t["max"],
            "step": t["step"],
            "value": t["value"][1],
        }
        self.slider_params = slider_params_out

    def set_numeric_f_params(
        self, f_params_num: Dict[str, float], t_span: Tuple[float, float]
    ):
        self.f_params_num = f_params_num
        self.t_span = t_span

    @staticmethod
    def _make_sliders(
        params: Dict[str, Dict[str, float]],
    ) -> Dict[str, widgets.FloatSlider]:
        sliders = {}
        for item in params.items():
            sliders[item[0]] = widgets.FloatSlider(**item[1])
        return sliders

    def model_curve(self):
        # close previous instance of this figure, if it exists
        # the name should be unique
        figname = "model_curve"
        plt.close(figname)

        fig, ax = plt.subplots(num=figname)
        fig.set_size_inches(6, 6)
        ax.set_aspect("equal", adjustable="datalim")
        fig.canvas.header_visible = False

        lines = plt.plot(0, 0)
        plt.ion()

        def plot(**params):
            # split params into groups
            # function params, that is all but t_0, t_n
            f_params = {
                key: params[key] for key in params.keys() if key not in ("t_0", "t_n")
            }
            # t_0, t_n
            t_span = {
                key: params[key] for key in params.keys() if key in ("t_0", "t_n")
            }

            t = np.linspace(
                t_span["t_0"],
                t_span["t_n"],
                int((t_span["t_n"] - t_span["t_0"]) * 100 + 1),
            )

            # this is order-sensitive
            x = self.x_f(t, f_params.values())
            y = self.y_f(t, f_params.values())
            self.set_numeric_f_params(f_params, tuple(t_span.values()))

            lines[0].set_data(x, y)
            ax.relim()
            ax.autoscale_view()

        # this ensures t_n isn't smaller than t_0
        def update_t_n_range(*args):
            sliders["t_n"].min = sliders["t_0"].value + sliders["t_n"].step

        sliders = self._make_sliders(self.slider_params)
        sliders["t_0"].observe(update_t_n_range, "value")
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
