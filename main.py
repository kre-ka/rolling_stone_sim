from sim import LinearSlope, QuadraticSlope, plot_sim_results


if __name__ == '__main__':
    # create a simulator instance
    sim = LinearSlope()
    # set slope coefficients
    sim.set_slope_coef(2.0, 6.0)
    # simulate movement
    t, x, y, tf = sim.simulate()

    # sim = QuadraticSlope()
    # sim.set_slope_coef(5, 10, 0.6)
    # t, x, y, tf = sim.simulate()

    plot_sim_results(t, x, y)