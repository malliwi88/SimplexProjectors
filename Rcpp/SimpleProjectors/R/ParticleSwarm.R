


setClass("ParticleSwarm",
         representation(size = "numeric",
                        swarm = "list",
                        initSwarm = "integer",
                        getGBest = "integer",
                        updateSwarm = "integer",
                        updateVelocities = "integer",
                        updatePositions = "integer",
                        problem = "Problem"),

         prototype(size = 55))


setGeneric("initSwarm", function(pso) standardGeneric("initSwarm"))
setMethod("initSwarm", signature("ParticleSwarm"),
          function(pso) {
            dimension <- pso@problem@dimension
            for (ix in 1:pso@size) {
              pso@swarm[[ix]] <- new("Particle",
                                     problem = pso@problem,
                                     dimension = dimension)

              pso@swarm[[ix]]@position <- runif(dimension)
              pso@swarm[[ix]] <- satisfyConstraints(pso@swarm[[ix]])
              pso@swarm[[ix]]@pbest <- pso@swarm[[ix]]@position
              pso@swarm[[ix]]@veloc <- runif(dimension, min = 0.001, max = 0.01)
            }
          })


setGeneric("getGBest", function(pso, parallel) standardGeneric("getGBest"))
setMethod("getGBest", signature("ParticleSwarm", "logical"),
          function(pso, parallel = FALSE) {
            if (parallel) {
              stop("Parallel getGBest not implemented yet")
            } else {
              gbest.ix <- 1
              gbest <- pso@swarm[[1]]
              gbest.fitness <- getParticleFitness(gbest)

              for (ix in 2:pso@size) {
                ix.fitness <- getParticleFitness(pso@swarm[[ix]])
                if (ix.fitness > gbest.fitness) {
                  gbest.ix <- ix
                  gbest <- pso@swarm[[ix]]
                  gbest.fitness <- ix.fitness
                }
              }
              return(gbest)
            }
          })


