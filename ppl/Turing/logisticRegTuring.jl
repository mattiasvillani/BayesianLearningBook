using Turing, StatsPlots, Random, LinearAlgebra
using StatsFuns: logistic

# Seeds data
y = [10, 23, 23, 26, 17, 5, 53, 55, 32, 46, 10, 8, 10, 8, 23, 0, 3, 22, 15, 32, 3]
N = [39, 62, 81, 51, 39, 6, 74, 72, 51, 79, 13, 16, 30, 28, 45, 4, 12, 41, 30, 51, 7]
x1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
x2 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

n = length(y)
X = [ones(n) x1 x2]
n,p = size(X)

# logit(z) = 1/(1+exp(-z)) # defining the logistic function

# Declare the Turing model:
@model function binomLogisticReg(y, N, X, τ)
    p = size(X,2)
    β ~ MvNormal(zeros(p),τ^2*I(p))  # prior
    Xβ = X*β
    n = length(y)  # number of observations
    for i in 1:n
        y[i] ~ Binomial(N[i],logistic(Xβ[i])) # model
    end
end

τ = 1    # Prior standard deviation
α = 0.40 # target acceptance probability in No U-Turn sampler
niter = 10000
nburn = 1000
postdraws = sample(binomLogisticReg(y, N, X, τ), NUTS(α), niter, discard_initial = nburn)
postdraws = sample(binomLogisticReg(y, N, X, τ), RWMH(MvNormal(p,1)), niter, discard_initial = nburn)
# Print an plot results
display(postdraws)
plot(postdraws)