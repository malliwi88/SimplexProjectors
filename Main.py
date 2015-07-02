__author__ = 'Stuart Gordon Reid'


from AssetSimulator import AssetSimulator
from Barebones import BarebonesOptimizer
from Portfolio import Portfolio
import pandas
import numpy

from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm



def plot(r, sim):
    asset_prices = sim.asset_prices(n, returns=r)
    for p in asset_prices:
        plt.plot(p)
    plt.show()


def fitness_landscape(asset_returns, corr, size):
    m_lagrange = numpy.zeros(shape=(size, size))
    m_repair = numpy.zeros(shape=(size, size))
    m_none = numpy.zeros(shape=(size, size))
    for i in range(size):
        for j in range(size):
            x, y = float(i/size), float(j/size)
            s = max(x + y, 0.001)
            x_r, y_r = x / s, y / s
            p = Portfolio(asset_returns, corr, numpy.array([x, y]))
            p_r = Portfolio(asset_returns, corr, numpy.array([x_r, y_r]))
            m_none[i][j] = p.min_objective()
            m_repair[i][j] = p_r.min_objective()
            m_lagrange[i][j] = p.lagrange_objective()
    plt.matshow(numpy.fliplr(m_none))
    plt.matshow(numpy.fliplr(m_repair))
    plt.matshow(numpy.fliplr(m_lagrange))
    plt.show()


def surface_plot(asset_returns, corr, size):
    X = numpy.arange(0.0, 5, float(5/size))
    Y = numpy.arange(0.0, 5, float(5/size))

    Z_none = numpy.zeros(shape=(size, size))
    Z_repair = numpy.zeros(shape=(size, size))
    Z_lagrange = numpy.zeros(shape=(size, size))
    for i in range(len(X)):
        for j in range(len(Y)):
            x, y = X[i], Y[j]
            p = Portfolio(asset_returns, corr, numpy.array([x, y]))
            Z_none[i][j] = p.min_objective()
            Z_lagrange[i][j] = p.lagrange_objective()
            Z_repair[i][j] = p.repair_objective()

    X, Y = numpy.meshgrid(X, Y)
    plot_s(X, Y, Z_none)
    plot_s(X, Y, Z_repair)
    plot_s(X, Y, Z_lagrange)


def plot_s(X, Y, Z):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


if __name__ == '__main__':
    n, sigma, delta, mu, time = 25, 0.125, float(1/252), 0.08, 500
    asset_simulator = AssetSimulator(delta, sigma, mu, time)
    asset_returns = asset_simulator.assets_returns(n)
    corr = pandas.DataFrame(asset_returns).transpose().corr()

    # surface_plot(asset_returns, corr, 100)
    # portfolio = Portfolio(asset_returns, corr)
    # plot(asset_returns, asset_simulator)

    pso = None
    pso = BarebonesOptimizer(25, asset_returns, corr)
    pso.optimize_lagrange(1001)

    pso = None
    pso = BarebonesOptimizer(25, asset_returns, corr)
    pso.optimize_preserving(1001)

    pso = None
    pso = BarebonesOptimizer(25, asset_returns, corr)
    pso.optimize_repair(1001)