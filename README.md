# rolling_stone_sim
### AKA roller coaster simulator
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/kre-ka/rolling_stone_sim/main?labpath=main.ipynb)

A fruit of my desire to learn some numeric simulations. Heavy use of [sympy](https://github.com/sympy/sympy) - who knows, maybe that's kind of a new thing to do (not that I checked). See [main.ipynb](main.ipynb) for how to use, or just try it out in [binder](https://mybinder.org/v2/gh/kre-ka/rolling_stone_sim/main?labpath=main.ipynb) (however animation is not really that smooth there).

![sim_result_demo](media/sim_result_demo.gif)

## Requirements
- Python (made with 3.11)
- everything else - see [requirements.txt](requirements.txt)

## What can I do with it?
- simulate a particle mass in gravity field traversing numerous parametric curves (which means ***it can do loops***)
- watch it move in *almost* real time (or slower/faster if you want to)
- try and evaluate different numeric ODE (Ordinary Differential Equation) solvers - well, the ones that [scipy](https://github.com/scipy/scipy) provides
- define your own curve using [sympy](https://github.com/sympy/sympy) expressions
- model a curve using interactive plot
## TODO
- add spline example(s) (there are only one-segment curves now)
- Bézier curve and spline modeler
- cover simulation runtime warnings
- add damping
- optimize $a(p)$ and $t(p)$ functions used for simulation
- evaluate simulation time
- simulate movement on animation run (for infinite simulation)
- allow path looping (like, path is a whole closed circle, and you can have this discontinuous jump from $p_n$ to $p_0$ (assuming initial velocity is non-zero))
- custom symplectic solver (maybe?)
- all the other TODO points that don't exist yet. The road never ends