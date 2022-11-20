# Using Rstan to fit the iid normal model on the internet speed data from the 
# book 'Bayesian Learning' by Mattias Villani (https://mattiasvillani.com)
# Book: https://github.com/mattiasvillani/BayesianLearningBook)

# Loading Stan and setting up nicer colours
library(rstan)
library(RColorBrewer)
plotColors = brewer.pal(12, "Paired")

# Define the Stan model
stanModelNormal = '
// The input data is a vector y of length N.
data {
  // data
  int<lower=0> N;
  vector[N] y;
  // prior
  real mu0;
  real<lower=0> kappa0;
  real<lower=0> nu0;
  real<lower=0> sigma20;
}

// The parameters in the model
parameters {
  real theta;
  real<lower=0> sigma2;
}

model {
  sigma2 ~ scaled_inv_chi_square(nu0, sqrt(sigma20));
  theta ~ normal(mu0,sqrt(sigma2/kappa0));
  y ~ normal(theta, sqrt(sigma2));
}
'

# Set up the observed data
data <- list(N = 5, y = c(15.77, 20.5, 8.26, 14.37, 21.09))

# Set up the prior
prior <- list(mu0 = 20, kappa0 = 1, nu0 = 5, sigma20 = 5^2)

# Sample from posterior using HMC
fit <- stan(model_code = stanModelNormal, data = c(data,prior), iter = 10000 )

# print and plot results
print(fit, pars = c("theta","sigma2"), probs=c(.1,.5,.9))
pairs(fit)
traceplot(fit, pars = c("theta", "sigma2"), nrow = 2)

# Extract the posterior samples from stan's fit object
postDraws <- extract(fit, permuted = TRUE) # return a list of arrays 
thetaDraws <- postDraws$theta
sigmaDraws <- sqrt(postDraws$sigma2)
cvDraws <- sigmaDraws/thetaDraws

# Plot marginals to reproduce the figures in the book
par(mfrow=c(2,2))
hist(thetaDraws, 30, main = expression(theta), xlab = "", ylab = "", yaxt='n', 
     col = plotColors[2], border = F)
hist(sigmaDraws^2, 70, main = expression(sigma^2), xlab = "", ylab = "", yaxt='n', 
     col = plotColors[2], border = F, xlim = c(0,100))
hist(sigmaDraws, 30, main = expression(sigma), xlab = "", ylab = "", yaxt='n', 
     col = plotColors[2], border = F)
hist(cvDraws, 30, main = expression(sigma/theta), xlab = "", ylab = "", yaxt='n', 
     col = plotColors[2], border = F)

