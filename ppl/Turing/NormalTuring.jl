# Turing.jl Bayesian analysis of Gaussian model with conjugate prior:
# x₁,...xₙ | θ, σ² ~ N(θ,σ²) using the conjugate prior
# σ² ~ Inv-χ²(ν₀,σ²₀)
# θ | σ² ~ N(μ₀,σ²/κ₀)

# Written for the book Bayesian Learning by Mattias Villani available at:
# https://github.com/mattiasvillani/BayesianLearningBook

using Turing, StatsPlots, Random
ScaledInverseChiSq(ν,τ²) = InverseGamma(ν/2,ν*τ²/2) # Inv-χ² distribution

# Setting up the Turing model:
@model function iidnormal(x, μ₀, κ₀, ν₀, σ²₀)
    σ² ~ ScaledInverseChiSq(ν₀, σ²₀)
    θ ~ Normal(μ₀,σ²/κ₀)  # prior
    n = length(x)  # number of observations
    for i in 1:n
        x[i] ~ Normal(θ, √σ²) # model
    end
end

# Set up the observed data
x = [15.77,20.5,8.26,14.37,21.09]

# Set up the prior
μ₀ = 20; κ₀ = 1; ν₀ = 5; σ²₀ = 5^2

# Settings of the Hamiltonian Monte Carlo (HMC) sampler.
niter = 10000
nburn = 1000
ϵ = 0.1
τ = 50
postdraws = sample(iidnormal(x, μ₀, κ₀, ν₀, σ²₀), HMC(ϵ, τ), niter, discard_initial = nburn)

# Print an plot results
display(postdraws)
plot(postdraws)

# Sample the posterior using HMC with NUTS
α = 0.65 # target acceptance probability in No U-Turn sampler
postdraws = sample(iidnormal(x, μ₀, κ₀, ν₀, σ²₀), NUTS(α), niter, discard_initial = nburn)

# Print an plot results
display(postdraws)
plot(postdraws)