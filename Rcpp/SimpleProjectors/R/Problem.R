

setClass("Problem",
         representation(dimension = "integer",
                        getFitness = "integer",
                        getViolations = "integer",
                        satisfyProblem = "integer"),
         prototype())


setGeneric("getFitness", function(problem, solution) standardGeneric("getFitness"))
setMethod("getFitness", signature("Problem", "numeric"),
          function(problem, solution) {

          })


setGeneric("getViolations", function(problem, solution) standardGeneric("getViolations"))
setMethod("getViolations", signature("Problem", "numeric"),
          function(problem, solution) {

          })


setGeneric("satisfyProblem", function(problem, solution) standardGeneric("satisfyProblem"))
setMethod("satisfyProblem", signature("Problem", "numeric"),
          function(problem, solution) {

          })