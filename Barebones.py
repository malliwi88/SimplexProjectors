__author__ = 'Stuart Gordon Reid'

"""
This file contains an implementation of the Barebones Particle Swarm Optimization algorithm for the Portfolio
Optimization problem. This file contains two constraint satisfaction techniques namely, repair by normalization and
lagrange multipliers method. These results are compared to the implementation of the Preserving Feasibility Barebones
Particle Swarm Optimization algorithm which uses a preserving feasibility constraint satisfaction technique.
"""


import numpy
import random
import numpy.random as nrand
from Portfolio import Portfolio


class BarebonesOptimizer:
    def __init__(self, swarm_size, asset_returns, corr):
        nrand.seed(seed=123456)
        self.corr = corr
        self.n = len(asset_returns)
        self.returns = asset_returns
        self.swarm_size = swarm_size
        self.swarm = []
        for i in range(swarm_size):
            ones = numpy.ones(self.n)
            self.swarm.append(nrand.dirichlet(ones, 1)[0])

    def optimize_repair(self, iterations):
        for i in range(iterations):
            best_index = 0
            best_weights = None
            best_fitness = float('+inf')
            for j in range(self.swarm_size):
                portfolio = Portfolio(self.returns, self.corr, self.swarm[j])
                fitness = portfolio.repair_objective()
                self.swarm[j] = portfolio.weights
                # print(numpy.sum(self.swarm[j]), list(self.swarm[j]))
                if fitness < best_fitness:
                    best_weights = self.swarm[j]
                    best_fitness = fitness
                    best_index = j
            for j in range(self.swarm_size):
                if j != best_index:
                    loc = numpy.array((best_weights + self.swarm[j]) / 2.0)
                    velocity = numpy.empty(len(best_weights))
                    for k in range(len(best_weights)):
                        velocity[k] = random.normalvariate(loc[k], 0.05)
                    self.swarm[j] += velocity
            if i % 25 == 0:
                print("Repair", i, best_fitness, numpy.sum(best_weights))

    def optimize_lagrange(self, iterations):
        for i in range(iterations):
            best_index = 0
            best_weights = None
            best_fitness = float('+inf')
            for j in range(self.swarm_size):
                portfolio = Portfolio(self.returns, self.corr, self.swarm[j])
                fitness = portfolio.lagrange_objective()
                if fitness < best_fitness:
                    best_weights = self.swarm[j]
                    best_fitness = fitness
                    best_index = j
            for j in range(self.swarm_size):
                if j != best_index:
                    loc = numpy.array((best_weights + self.swarm[j]) / 2.0)
                    velocity = numpy.empty(len(best_weights))
                    for k in range(len(best_weights)):
                        velocity[k] = random.normalvariate(loc[k], 0.05)
                    self.swarm[j] = velocity
            if i % 25 == 0:
                print("Lagrange", i, best_fitness, numpy.sum(best_weights))

    def optimize_preserving(self, iterations):
        for i in range(iterations):
            best_index = 0
            best_weights = None
            best_fitness = float('+inf')
            for j in range(self.swarm_size):
                portfolio = Portfolio(self.returns, self.corr, self.swarm[j])
                fitness = portfolio.min_objective()
                if fitness < best_fitness:
                    best_weights = self.swarm[j]
                    best_fitness = fitness
                    best_index = j
            for j in range(self.swarm_size):
                if j != best_index:
                    r = nrand.dirichlet(numpy.ones(self.n), 1)[0] - float(1/self.n)
                    velocity = numpy.array(best_weights - self.swarm[j]) + r
                    self.swarm[j] += velocity
            if i % 25 == 0:
                print("Preserving", i, best_fitness, numpy.sum(best_weights))