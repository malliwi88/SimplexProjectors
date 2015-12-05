

setClass("Particle",
         representation(c1 = "numeric",
                        c2 = "numeric",
                        w = "numeric",
                        problem = "Problem",

                        dimension = "numeric",
                        solution = "numeric",
                        pbest = "numeric",
                        veloc = "numeric",

                        updateVelocity = "integer",
                        updatePosition = "integer",
                        satisfyConstraints = "integer",

                        getParticleFitness = "integer",
                        getParticleViolations = "integer"),

         prototype(c1 = 1.4,
                   c2 = 1.4,
                   w = 0.7))


setGeneric("updateVelocity", function(particle, gbest) standardGeneric("updateVelocity"))
setMethod("updateVelocity", signature("Particle", "Particle"),
          function(particle, gbest) {
            r1 <- runif(particle@dimension)
            r2 <- runif(particle@dimension)

            gbest.diff <- particle@solution - gbest@pbest
            pbest.diff <- particle@solution - particle@pbest

            social <- particle@c2 * r2 * gbest.diff
            cognitive <- particle@c1 * r1 * pbest.diff

            particle@veloc <- particle@w * particle@veloc + social + cognitive
            return(particle)
          })


setGeneric("updatePosition", function(particle) standardGeneric("updatePosition"))
setMethod("updatePosition", signature("Particle"),
          function(particle) {
            fitness.pre <- getParticleFitness(particle@problem, particle)
            particle@solution <- particle@solution + particle@veloc
            fitness.post <- getParticleFitness(particle@problem, particle)

            # Update the personal best position.
            if (fitness.post > fitness.pre)
              particle@pbest <- particle@solution

            return(particle)
          })


setGeneric("satisfyConstraints", function(particle) standardGeneric("satisfyConstraints"))
setMethod("satisfyConstraints", signature("Particle"),
          function(particle) {
            return(satisfyProblem(particle@problem, particle@solution))
          })


setGeneric("getParticleFitness", function(particle) standardGeneric("getParticleFitness"))
setMethod("getParticleFitness", signature("Particle"),
          function(particle) {
            return(getFitness(particle@problem, particle@solution))
          })


setGeneric("getParticleViolations", function(particle) standardGeneric("getParticleViolations"))
setMethod("getParticleViolations", signature("Particle"),
          function(particle) {
            return(getViolations(particle@problem, particle@solution))
          })
