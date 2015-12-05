

setClass("Portfolio",
         representation(expected.returns = "numeric",
                        expected.risks = "numeric",
                        expected.corr = "matrix",

                        allocation.max = "numeric",
                        allocation.min = "numeric",
                        allocation.sum = "numeric",

                        getPortfolioReturn = "integer",
                        getPortfolioViolations = "integer"),

         prototype(allocation.max = 1.0,
                   allocation.min = 0.0,
                   allocation.sum = 1.0),

         contains = "Problem")


setMethod("getFitness", signature("Portfolio", "numeric"),
          function(problem, solution) {
            return(getPortfolioReturn(problem, solution))
          })


setGeneric("getPortfolioReturn", function(portfolio, weights) standardGeneric("getPortfolioReturn"))
setMethod("getPortfolioReturn", signature("Portfolio", "numeric"),
          function(portfolio, weights) {

          })


setMethod("getViolations", signature("Portfolio", "numeric"),
          function(problem, solution) {
            return(getPortfolioViolations(problem, solution))
          })


setGeneric("getPortfolioViolations", function(portfolio, weights) standardGeneric("getPortfolioViolations"))
setMethod("getPortfolioViolations", signature("Portfolio", "numeric"),
          function(portfolio, weights) {

          })
