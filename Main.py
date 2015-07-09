__author__ = 'Stuart Gordon Reid'


from AssetSimulator import AssetSimulator
from Barebones import BarebonesOptimizer
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import Portfolio
import cProfile
import pandas
import numpy
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def plot_paths(r, sim):
    """
    This method plots a number of asset paths generated using Geometric Brownian Motion
    """
    asset_prices = sim.asset_prices(n, returns=r)
    for p in asset_prices:
        plt.plot(p)
    plt.show()


def plot_results(results, labels, ylabel):
    plt.style.use("bmh")
    plt.ylabel(ylabel)
    plt.xlabel("Iterations")
    linestyles = ['-', '--']
    for i in range(len(results)):
        if labels is None:
            plt.plot(results[i])
        else:
            plt.plot(results[i], label=labels[i], linestyle=linestyles[i % 2], linewidth=2)
    plt.legend(loc="best")
    plt.show()


def two_dimensional_landscape(returns, corr_m, size):
    """
    This method plots the fitness landscape for each three methods in two dimensions (grid)
    """
    m_lagrange = numpy.zeros(shape=(size, size))
    m_repair = numpy.zeros(shape=(size, size))
    m_none = numpy.zeros(shape=(size, size))
    for i in range(size):
        for j in range(size):
            x, y = float(i/size), float(j/size)
            p = Portfolio.Portfolio(returns, corr_m, numpy.array([x, y]))
            m_lagrange[i][j] = p.lagrange_objective()
            m_none[i][j] = p.min_objective()
            m_repair[i][j] = p.repair_objective()
    plt.matshow(numpy.fliplr(m_none))
    plt.matshow(numpy.fliplr(m_repair))
    plt.matshow(numpy.fliplr(m_lagrange))
    plt.show()


def three_dimensional_landscape(returns, corr_m, size, c_e=1.0, c_b=1.0, m_e=1.0, m_b=1.0):
    """
    This method plots the fitness landscape for each three methods in three dimensions (surface plot)
    """
    step = float(1/size)
    x_axis = numpy.arange(0.0, 1.0, step)
    y_axis = numpy.arange(0.0, 1.0, step)
    z_axis_nochange = numpy.zeros(shape=(size, size))
    z_axis_repaired = numpy.zeros(shape=(size, size))
    z_axis_penaltym = numpy.zeros(shape=(size, size))
    z_axis_lagrange = numpy.zeros(shape=(size, size))
    for i in range(size):
        for j in range(size):
            x, y = x_axis[i], y_axis[j]
            p = Portfolio.Portfolio(returns, corr_m, numpy.array([x, y]))
            z_axis_nochange[i][j] = p.min_objective()
            z_axis_penaltym[i][j] = p.penalty_objective(c_e, c_b)
            z_axis_lagrange[i][j] = p.lagrange_objective(c_e, c_b, m_e, m_b)
            z_axis_repaired[i][j] = p.repair_objective()
    x_axis, y_axis = numpy.meshgrid(x_axis, y_axis)
    plot_surface(x_axis, y_axis, z_axis_nochange, "N")
    plot_surface(x_axis, y_axis, z_axis_repaired, "R")
    plot_surface(x_axis, y_axis, z_axis_penaltym, "P")
    plot_surface(x_axis, y_axis, z_axis_lagrange, "L")


def plot_surface(X, Y, Z, label):
    """
    This method actually plots and shows the three dimensional surface
    """
    fig = plt.figure(figsize=(15, 10))
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)
    '''
    for ii in range(0, 360, 90):
        ax.view_init(elev=10., azim=ii)
        plt.savefig("Views/View II " + label + " " + str(ii) + ".png")
    '''
    plt.show()


def runner(n, sigma, delta, mu, time, iterations, simulations):
    asset_simulator = AssetSimulator(delta, sigma, mu, time)

    Portfolio.memoizer = {}
    none, penalty, lagrange, repair, preserve, ss = [], [], [], [], [], 25
    none_ve, penalty_ve, lagrange_ve, repair_ve, preserve_ve = [], [], [], [], []
    none_vb, penalty_vb, lagrange_vb, repair_vb, preserve_vb = [], [], [], [], []
    ce, cb, le, lb = 2.0, 2.0, 0.5, 0.5

    for i in range(simulations):
        print("Iteration", i, "Simulating Returns ...")
        asset_returns = asset_simulator.assets_returns(n)
        corr = pandas.DataFrame(asset_returns).transpose().corr()
        # three_dimensional_landscape(asset_returns, corr, 100)

        print("Iteration", i, "Starting Optimizer ...")
        none_opt = BarebonesOptimizer(ss, asset_returns, corr)
        result, violation_e, violation_b = none_opt.optimize_none(iterations+1, ce, cb, le, lb)
        none_ve.append(violation_e)
        none_vb.append(violation_b)
        none.append(result)

        print("Iteration", i, "Starting Penalty Optimizer ...")
        lagrange_opt = BarebonesOptimizer(ss, asset_returns, corr)
        result, violation_e, violation_b = lagrange_opt.optimize_penalty(iterations+1, ce, cb, le, lb)
        penalty_ve.append(violation_e)
        penalty_vb.append(violation_b)
        penalty.append(result)

        print("Iteration", i, "Starting Lagrange Optimizer ...")
        lagrange_opt = BarebonesOptimizer(ss, asset_returns, corr)
        result, violation_e, violation_b = lagrange_opt.optimize_lagrange(iterations+1, ce, cb, le, lb)
        lagrange_ve.append(violation_e)
        lagrange_vb.append(violation_b)
        lagrange.append(result)

        print("Iteration", i, "Starting Repair Method Optimizer ...")
        repair_opt = BarebonesOptimizer(ss, asset_returns, corr)
        result, violation_e, violation_b = repair_opt.optimize_repair(iterations+1, ce, cb, le, lb)
        repair_ve.append(violation_e)
        repair_vb.append(violation_b)
        repair.append(result)

        print("Iteration", i, "Starting Preserving Feasibility Optimizer ...")
        preserve_opt = BarebonesOptimizer(ss, asset_returns, corr)
        result, violation_e, violation_b = preserve_opt.optimize_preserving(iterations+1, ce, cb, le, lb)
        preserve_ve.append(violation_e)
        preserve_vb.append(violation_b)
        preserve.append(result)

    n_r, n_ve, n_vb = pandas.DataFrame(none), pandas.DataFrame(none_ve), pandas.DataFrame(none_vb)
    r_r, r_ve, r_vb = pandas.DataFrame(repair), pandas.DataFrame(repair_ve), pandas.DataFrame(repair_vb)
    p_r, p_ve, p_vb = pandas.DataFrame(preserve), pandas.DataFrame(preserve_ve), pandas.DataFrame(preserve_vb)
    pr_r, pr_ve, pr_vb = pandas.DataFrame(penalty), pandas.DataFrame(penalty_ve), pandas.DataFrame(penalty_vb)
    l_r, l_ve, l_vb = pandas.DataFrame(lagrange), pandas.DataFrame(lagrange_ve), pandas.DataFrame(lagrange_vb)

    n_r.to_csv("Results/None Fitness.csv")
    n_ve.to_csv("Results/None Equality.csv")
    n_vb.to_csv("Results/None Boundary.csv")

    r_r.to_csv("Results/Repair Fitness.csv")
    r_ve.to_csv("Results/Repair Equality.csv")
    r_vb.to_csv("Results/Repair Boundary.csv")

    p_r.to_csv("Results/Preserve Fitness.csv")
    p_ve.to_csv("Results/Preserve Equality.csv")
    p_vb.to_csv("Results/Preserve Boundary.csv")

    pr_r.to_csv("Results/Penalty Fitness.csv")
    pr_ve.to_csv("Results/Penalty Equality.csv")
    pr_vb.to_csv("Results/Penalty Boundary.csv")

    l_r.to_csv("Results/Lagrangian Fitness.csv")
    l_ve.to_csv("Results/Lagrangian Equality.csv")
    l_vb.to_csv("Results/Lagrangian Boundary.csv")

    plot_results([n_r.mean(), r_r.mean(), pr_r.mean(), l_r.mean(), p_r.mean()],
                 ["Algorithm 1", "Algorithm 2", "Algorithm 3", "Algorithm 4", "Algorithm 5"],
                 "Average Fitness")

    plot_results([n_r.std(), r_r.std(), pr_r.std(), l_r.std(), p_r.std()],
                 ["Algorithm 1", "Algorithm 2", "Algorithm 3", "Algorithm 4", "Algorithm 5"],
                 "Fitness Standard Deviation")

    plot_results([r_r.mean(), pr_r.mean(), l_r.mean(), p_r.mean()],
                 ["Algorithm 2", "Algorithm 3", "Algorithm 4", "Algorithm 5"],
                 "Average Fitness")

    plot_results([r_r.std(), pr_r.std(), l_r.std(), p_r.std()],
                 ["Algorithm 2", "Algorithm 3", "Algorithm 4", "Algorithm 5"],
                 "Fitness Standard Deviation")

    plot_results([n_ve.mean(), r_ve.mean(), pr_ve.mean(), l_ve.mean(), p_ve.mean()],
                 ["Algorithm 1", "Algorithm 2", "Algorithm 3", "Algorithm 4", "Algorithm 5"],
                 "Average Equality Constraint Violation")

    plot_results([n_ve.std(), r_ve.std(), pr_ve.std(), l_ve.std(), p_ve.std()],
                 ["Algorithm 1", "Algorithm 2", "Algorithm 3", "Algorithm 4", "Algorithm 5"],
                 "Equality Constraint Violation Standard Deviation")

    plot_results([n_vb.mean(), r_vb.mean(), pr_vb.mean(), l_vb.mean(), p_vb.mean()],
                 ["Algorithm 1", "Algorithm 2", "Algorithm 3", "Algorithm 4", "Algorithm 5"],
                 "Average Boundary Constraint Violation")

    plot_results([n_vb.std(), r_vb.std(), pr_vb.std(), l_vb.std(), p_vb.std()],
                 ["Algorithm 1", "Algorithm 2", "Algorithm 3", "Algorithm 4", "Algorithm 5"],
                 "Boundary Constraint Violation Standard Deviation")

    plot_results([r_ve.mean(), pr_ve.mean(), l_ve.mean(), p_ve.mean()],
                 ["Algorithm 2", "Algorithm 3", "Algorithm 4", "Algorithm 5"],
                 "Average Equality Constraint Violation")

    plot_results([r_ve.std(), pr_ve.std(), l_ve.std(), p_ve.std()],
                 ["Algorithm 2", "Algorithm 3", "Algorithm 4", "Algorithm 5"],
                 "Equality Constraint Violation Standard Deviation")

    plot_results([r_vb.mean(), pr_vb.mean(), l_vb.mean(), p_vb.mean()],
                 ["Algorithm 2", "Algorithm 3", "Algorithm 4", "Algorithm 5"],
                 "Average Boundary Constraint Violation")

    plot_results([r_vb.std(), pr_vb.std(), l_vb.std(), p_vb.std()],
                 ["Algorithm 2", "Algorithm 3", "Algorithm 4", "Algorithm 5"],
                 "Boundary Constraint Violation Standard Deviation")


def surface_plotter(n, sigma, delta, mu, time, c_e, c_b, m_e, m_b):
    asset_simulator = AssetSimulator(delta, sigma, mu, time)
    asset_returns = asset_simulator.assets_returns(n)
    corr = pandas.DataFrame(asset_returns).transpose().corr()
    three_dimensional_landscape(asset_returns, corr, 200, c_e, c_b, m_e, m_b)


if __name__ == '__main__':
    runner(15, 0.125, float(1/252), 0.08, 500, 100, 55)
    # surface_plotter(2, 0.125, float(1/252), 0.08, 250, 2.0, 2.0, -20.0, -20.0)
