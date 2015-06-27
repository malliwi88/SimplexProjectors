__author__ = 'Stuart Gordon Reid'


from AssetSimulator import AssetSimulator
from Barebones import BarebonesOptimizer
from Portfolio import Portfolio
import matplotlib.pylab as plt
import numpy.random as nrand
import pandas
import numpy


def plot(r, sim):
    asset_prices = sim.asset_prices(n, returns=r)
    for p in asset_prices:
        plt.plot(p)
    plt.show()

if __name__ == '__main__':
    n, sigma, delta, mu, time = 15, 0.125, float(1/252), 0.08, 500
    asset_simulator = AssetSimulator(delta, sigma, mu, time)
    asset_returns = asset_simulator.assets_returns(n)
    corr = pandas.DataFrame(asset_returns).transpose().corr()
    portfolio = Portfolio(asset_returns, corr)
    # plot(asset_returns, asset_simulator)

    pso = None
    pso = BarebonesOptimizer(25, asset_returns, corr)
    pso.optimize_preserving(5001)

    pso = None
    pso = BarebonesOptimizer(25, asset_returns, corr)
    pso.optimize_repair(5001)

    pso = None
    pso = BarebonesOptimizer(25, asset_returns, corr)
    pso.optimize_lagrange(5001)