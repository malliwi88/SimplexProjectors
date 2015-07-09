__author__ = 'Stuart Gordon Reid'

"""
This file contains an implementation of a very simple portfolio optimization problem.
"""


import math
import numpy


memoize = False
memoizer = {}


class Portfolio:
    def __init__(self, asset_returns, corr, weights, r=3):
        """
        This method constructs a portfolio
        :param asset_returns: list of numpy vectors of historical / simulated returns
        :param corr: the correlation matrix of the returns
        :param weights: portfolio weights
        """
        self.asset_returns = asset_returns
        self.n = len(asset_returns)
        self.corr = corr
        # Calculate E(r)'s and sigma(r)'s
        self.expected_return = numpy.empty(self.n)
        self.expected_risk = numpy.empty(self.n)
        for i in range(self.n):
            # prices = returns_to_prices(asset_returns[i])
            # self.expected_return[i] = (prices[len(prices) - 1] / prices[0]) - 1
            self.expected_return[i] = numpy.mean(asset_returns[i])
            self.expected_risk[i] = numpy.std(asset_returns[i])
        # Construct the initial weights
        self.weights = numpy.around(weights, r)
        self.hash = hash(tuple(self.weights))

    def portfolio_return(self):
        """
        Calculates the expected return of the portfolio
        :return: expected return of the portfolio
        """
        return numpy.sum(numpy.dot(self.expected_return, self.weights))

    def portfolio_risk(self):
        """
        Calculates the expected risk of the portfolio
        :return: expected risk of the portfolio
        """
        variance = numpy.sum(numpy.dot(self.weights**2, self.expected_risk**2))
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    weights_ij = self.weights[i] * self.weights[j]
                    risks_ij = self.expected_risk[i] * self.expected_risk[j]
                    variance += weights_ij * risks_ij * self.corr[i][j]
        return math.sqrt(variance)

    def max_objective(self, risk_free_rate=0.00):
        """
        Returns the expected sharpe ratio
        :param risk_free_rate: the risk free rate
        :return: the expected sharpe ratio
        """
        return (self.portfolio_return() - risk_free_rate) / self.portfolio_risk()

    def min_objective(self):
        """
        Returns the negative of the max_objective
        :return: - expected sharpe ratio
        """
        if memoize:
            try:
                res = memoizer[self.hash]
                return res
            except KeyError:
                min_opt = -self.max_objective()
                memoizer[self.hash] = min_opt
                return min_opt
        else:
            return -self.max_objective()

    def penalty_objective(self, ce, cb):
        """
        Returns - expected sharpe ratio plus a weight penalty
        :return: the lagrange method objective
        """
        penalty_e = self.get_equality_penalty()
        penalty_b = self.get_boundary_penalty()
        fitness = self.min_objective()
        fitness += ce * penalty_e
        fitness += cb * penalty_b
        return fitness

    def lagrange_objective(self, ce, cb, le, lb):
        """
        Returns - expected sharpe ratio plus a weight penalty
        :return: the lagrange method objective
        """
        penalty_e = self.get_equality_penalty()
        penalty_b = self.get_boundary_penalty()
        fitness = self.min_objective()
        fitness += float(ce/2.0) * penalty_e
        fitness += float(cb/2.0) * penalty_b
        fitness -= le * penalty_e
        fitness -= lb * penalty_b
        return fitness

    def repair(self):
        """
        Repairs the portfolio weights using normalization by the sum of weights
        :return: the normalized / repaired portfolio weights
        """
        for w in range(len(self.weights)):
            self.weights[w] = max(self.weights[w], 0)
        self.weights /= numpy.sum(self.weights)

    def repair_objective(self):
        self.repair()
        return self.min_objective()

    def returns_to_prices(self, returns):
        prices = numpy.empty(len(returns) + 1)
        prices[0] = 100
        for i in range(len(returns)):
            prices[i + 1] = prices[i] * math.exp(returns[i])
        return prices

    def get_fitness(self, method, ce, cb, le, lb):
        fitness = self.min_objective()
        if method == "repair":
            fitness = self.repair_objective()
        if method == "penalty":
            fitness = self.penalty_objective(ce, cb)
        if method == "lagrange":
            fitness = self.lagrange_objective(ce, cb, le, lb)
        return fitness

    def get_equality_penalty(self):
        return math.pow(1.0 - float(numpy.sum(self.weights)), 2.0)

    def get_boundary_penalty(self):
        penalty = 0.0
        for w in self.weights:
            if w < 0.0:
                penalty += abs(w)
        return math.pow(penalty, 2.0)