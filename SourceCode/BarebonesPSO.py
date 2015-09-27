import os
import math
import numpy
import pandas
import random
import Quandl
import numpy.random as nrand


class BackTester:
    def __init__(self, returns, window, simplex_method):
        # Check that the returns are a data frame
        assert isinstance(returns, pandas.DataFrame)
        self.sm = simplex_method
        self.window = window
        self.returns = returns
        self.assets = returns.columns
        self.nassets = len(self.assets)
        self.all_days = len(returns[self.assets[0]])
        self.days = self.all_days - window
        self.portfolio_weights = []
        self.portfolio_returns = []

    def backtest(self):
        # for t in range(self.window, self.all_days):
        for t in range(self.window, self.window + 3):
            returns = self.returns[(t-self.window):t]
            bpso = BarebonesPSO(returns, method=self.sm)
            self.portfolio_weights.append(bpso.optimize())

            tth_returns = numpy.array(self.returns.iloc[[t]])[0]
            tth_weights = numpy.array(self.portfolio_weights[t - self.window])
            tth_portfolio_return = numpy.dot(tth_weights, tth_returns)
            self.portfolio_returns.append(tth_portfolio_return)

            print("Time =", t,
                  "\tr =", tth_portfolio_return,
                  "\tR =", list(tth_returns),
                  "\tW =", list(tth_weights))

        return self.portfolio_returns


class BarebonesPSO:
    def __init__(self, returns, method, swarm=30):
        self.returns = returns
        self.swarm = swarm
        self.corr = returns.corr()
        self.n = len(self.returns.columns)
        self.method = method

        self.pbest_portfolios = []
        self.xbest_portfolios = []
        for i in range(self.swarm):
            weights = nrand.dirichlet(numpy.ones(self.n), 1)[0]
            self.xbest_portfolios.append(Portfolio(self.returns, weights))
            self.pbest_portfolios.append(Portfolio(self.returns, weights))

    def optimize(self, niter=250, ce=2.0, cb=2.0, le=0.5, lb=0.5):
        if self.method == "none":
            return self.__updateswarm__(niter, "none", ce, cb, le, lb)
        elif self.method == "repair":
            return self.__updateswarm__(niter, "repair", ce, cb, le, lb)
        elif self.method == "penalty":
            return self.__updateswarm__(niter, "penalty", ce, cb, le, lb)
        elif self.method == "lagrange":
            return self.__updateswarm__(niter, "lagrange", ce, cb, le, lb)
        elif self.method == "preserve":
            return self.__updateswarm__(niter, "preserving", ce, cb, le, lb)

    def __bestparticle__(self, objective, ce, cb, le, lb):
        gbest_index, gbest_fitness = 0, float('+inf')

        for i in range(1, self.swarm):
            pbest_portfolio = self.pbest_portfolios[i]
            xbest_portfolio = self.xbest_portfolios[i]

            if objective == "repair":
                xbest_portfolio.repair()

            pbest_fitness = pbest_portfolio.get_fitness(objective, ce, cb, le, lb)
            xbest_fitness = xbest_portfolio.get_fitness(objective, ce, cb, le, lb)

            if xbest_fitness <= pbest_fitness:
                xweights = xbest_portfolio.get_weights()
                self.pbest_portfolios[i].update_weights(xweights)

            if xbest_fitness < gbest_fitness:
                gbest_index = i
                gbest_fitness = xbest_fitness

        return gbest_index, gbest_fitness

    def __updateswarm__(self, iterations, objective, ce, cb, le, lb, growth_rate=1.1):
        history = numpy.zeros(iterations)
        violation_e = numpy.zeros(iterations)
        violation_b = numpy.zeros(iterations)

        for i in range(iterations):
            best_index, best_fitness = self.__bestparticle__(objective, ce, cb, le, lb)
            best_weights = self.pbest_portfolios[best_index].get_weights()
            best_portfolio = self.pbest_portfolios[best_index]

            for j in range(self.swarm):
                if j != best_index:
                    if objective != "preserving":
                        pbest_weights = self.pbest_portfolios[j].get_weights()
                        loc = numpy.array((best_weights + pbest_weights) / 2.0)
                        sig = numpy.array(numpy.abs(best_weights - pbest_weights))
                        velocity = numpy.empty(len(best_weights))
                        for k in range(len(best_weights)):
                            velocity[k] = random.normalvariate(loc[k], sig[k])
                        self.xbest_portfolios[j].update_weights(velocity)

                    elif objective == "preserving":
                        pbest_weights = self.pbest_portfolios[j].get_weights()
                        loc = numpy.array((best_weights + pbest_weights)/2)
                        velocity = nrand.dirichlet(loc, 1)[0]
                        self.xbest_portfolios[j].update_weights(velocity)

            history[i] = best_portfolio.min_objective()
            violation_e[i] = best_portfolio.get_boundary_penalty()
            violation_b[i] = best_portfolio.get_equality_penalty()
            if objective == "penalty" or objective == "lagrange":
                ce *= growth_rate
                cb *= growth_rate
                if objective == "lagrange":
                    le -= ce * violation_e[i]
                    lb -= cb * violation_b[i]

        # return history, violation_e, violation_b
        bpi, bpf = self.__bestparticle__(objective, ce, cb, le, lb)
        return self.pbest_portfolios[bpi].weights


class Portfolio:
    def __init__(self, returns, weights):
        self.returns = returns
        self.weights = weights
        self.n = len(self.returns.columns)
        self.corr = self.returns.corr()
        self.expected_return = numpy.array(returns.mean())
        self.expected_risk = numpy.array(returns.std())

    def update_weights(self, weights):
        self.weights = weights

    def get_weights(self):
        return self.weights

    def portfolio_return(self):
        return numpy.sum(numpy.dot(self.expected_return, self.weights))

    def portfolio_risk(self):
        variance = 0.0
        for i in range(self.n):
            w = self.weights[i] * self.weights
            r = self.expected_risk[i] * self.expected_risk
            variance += numpy.sum(w * r * numpy.array(self.corr.iloc[[i]])[0])
        return math.sqrt(variance)

    def max_objective(self, risk_free_rate=0.00):
        return (self.portfolio_return() - risk_free_rate) / self.portfolio_risk()

    def min_objective(self):
        return -self.max_objective()

    def penalty_objective(self, ce, cb):
        penalty_e = self.get_equality_penalty()
        penalty_b = self.get_boundary_penalty()
        fitness = self.min_objective()
        fitness += ce * penalty_e
        fitness += cb * penalty_b
        return fitness

    def lagrange_objective(self, ce, cb, le, lb):
        penalty_e = self.get_equality_penalty()
        penalty_b = self.get_boundary_penalty()
        fitness = self.min_objective()
        fitness += float(ce/2.0) * penalty_e
        fitness += float(cb/2.0) * penalty_b
        fitness -= le * penalty_e
        fitness -= lb * penalty_b
        return fitness

    def repair(self):
        for w in range(len(self.weights)):
            self.weights[w] = max(self.weights[w], 0)
        self.weights /= numpy.sum(self.weights)

    def repair_objective(self):
        self.repair()
        return self.min_objective()

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


class QuandlInterface:
    def __init__(self, api_key):
        self.api_key = api_key

    def get_data_set(self, argument):
        file_name = argument.to_string()
        basepath = os.path.dirname(__file__)
        path = os.path.abspath(os.path.join(basepath, os.pardir, "MarketData", file_name))
        try:
            data_frame = pandas.read_csv(path)
            data_frame = data_frame.set_index("Date")
            return data_frame
        except:
            data_frame = self.download_data_set(argument)
            data_frame.to_csv(path, mode="w+")
            return data_frame

    def download_data_set(self, argument):
        assert isinstance(argument, Argument)
        data_frame = None
        try:
            data_set_name = argument.id
            if argument.prefix is not None:
                data_set_name = argument.prefix + data_set_name
            data_frame = Quandl.get(data_set_name, authtoken=self.api_key,
                                    trim_start=argument.start, trim_end=argument.end,
                                    transformation=argument.transformation, collapse=argument.collapse)
            assert isinstance(data_frame, pandas.DataFrame)
            for d in argument.drop:
                try:
                    data_frame = data_frame.drop(d, axis=1)
                except:
                    continue
        except Quandl.DatasetNotFound:
            print("Data set not found")
        except Quandl.ErrorDownloading:
            print("Error downloading")
        except Quandl.ParsingError:
            print("Parsing error")
        except Quandl.WrongFormat:
            print("Wrong format")
        except Quandl.CallLimitExceeded:
            print("Call limit exceeded")
        except Quandl.CodeFormatError:
            print("Code format error")
        except Quandl.MissingToken:
            print("Missing token")
        if data_frame is None:
            raise Exception("Data Set Not Initialized", argument.id)
        else:
            return data_frame

    def get_data_sets(self, arguments):
        # assert isinstance(arguments, [Argument])
        combined_data_frame = None
        for arg in arguments:
            assert isinstance(arg, Argument)
            arg_data_frame = self.get_data_set(arg)
            new_columns = []
            for i in range(len(arg_data_frame.columns)):
                new_columns.append(arg.id + "_" + arg_data_frame.columns[i])
            arg_data_frame.columns = new_columns
            if combined_data_frame is None:
                combined_data_frame = arg_data_frame
            else:
                combined_data_frame = combined_data_frame.join(arg_data_frame)
        combined_data_frame = combined_data_frame.dropna()
        return combined_data_frame


class Argument:
    def __init__(self, id, start, end, prefix=None, drop=None, rdiff="none", collapse="none"):
        self.id = id
        self.start = start
        self.end = end
        self.transformation = rdiff
        self.collapse = collapse
        self.prefix = prefix
        # The default drop columns for Google Finance data
        if drop is None:
            drop = ["High", "Low", "Open", "Volume", "Adjusted Close", ""]
        self.drop = drop

    def to_string(self):
        unique_id = "Cache"
        unique_id += " id=" + self.id
        unique_id += " start=" + self.start
        unique_id += " end=" + self.end
        unique_id += " trans=" + self.transformation
        unique_id += ".csv"
        return unique_id.replace("\\", "-").replace("/", "-")


if __name__ == '__main__':
    stocks = ["GOOG/JSE_SBK",
              "GOOG/JSE_ANG",
              "GOOG/JSE_SHP"]

    args = []
    for s in stocks:
        args.append(Argument(id=s, start="2010-01-01", end="2015-01-01", rdiff="rdiff"))

    qi = QuandlInterface("N9HccV672zuuiU5MUvcq")
    stock_data = qi.get_data_sets(args)

    lagrangian_back_tester = BackTester(stock_data, window=16, simplex_method="lagrange")
    lagrangian_returns = lagrangian_back_tester.backtest()

    print(lagrangian_returns)
