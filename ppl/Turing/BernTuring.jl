# Turing.jl Bayesian analysis of the Bernoulli model with the Beta prior: 
# y₁,...yₙ | θ ~ Bern(θ)
# θ ~ Beta(α,β)

# Written for the book Bayesian Learning by Mattias Villani available at:
# https://github.com/mattiasvillani/BayesianLearningBook

using Turing, StatsPlots, Random

# Declare the Turing model:
@model function iidbern(y, α, β)
    θ ~ Beta(α,β)  # prior
    N = length(y)  # number of observations
    for n in 1:N
        y[n] ~ Bernoulli(θ) # model
    end
end

# Set up the observed data
data = [0,1,1,0,0,1,1,0,1,1]

# Settings of the Hamiltonian Monte Carlo (HMC) sampler.
niter = 10000
nburn = 1000
ϵ = 0.1
τ = 10

# Sample the posterior using HMC
postdraws = sample(iidbern(data, 1, 2), HMC(ϵ, τ), niter, discard_initial = nburn)
plot(postdraws)

# Print an plot results
display(postdraws)
plot(postdraws)