__author__ = 'Stuart Gordon Reid'

"""
This file contains an implementation of a very simple portfolio optimization problem.
"""


import math
import numpy


class Portfolio:
    def __init__(self, asset_returns, corr, weights=None):
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
        if weights is None:
            self.weights = numpy.empty(self.n)
            equal_weight_const = float(1 / self.n)
            self.weights.fill(equal_weight_const)
        else:
            self.weights = weights

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
                weights_ij = self.weights[i] * self.weights[j]
                risks_ij = self.expected_risk[i] * self.expected_risk[j]
                variance += weights_ij * risks_ij * self.corr[i][j]
        return math.sqrt(variance)

    def repair(self):
        """
        Repairs the portfolio weights using normalization by the sum of weights
        :return: the normalized / repaired portfolio weights
        """
        sum_weights = float(numpy.sum(self.weights))
        sum_vector = numpy.empty(self.n)
        sum_vector.fill(sum_weights)
        self.weights = self.weights / sum_vector

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
        return -self.max_objective()

    def lagrange_objective(self):
        """
        Returns - expected sharpe ratio plus a weight penalty
        :return: the lagrange method objective
        """
        penalty = 1 - float(numpy.sum(self.weights))
        return self.min_objective() + numpy.abs(penalty)


def returns_to_prices(returns):
    prices = numpy.empty(len(returns) + 1)
    prices[0] = 100
    for i in range(len(returns)):
        prices[i + 1] = prices[i] * math.exp(returns[i])
    return prices