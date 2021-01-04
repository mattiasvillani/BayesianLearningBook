# Using Stan for optimizing the log-likelihood

# Set paths. Change this on your computer.
path2StanCode = "/home/mv/Dropbox/BayesBook/Code/PPL/Stan/"
setwd(path2StanCode)

# Better colors
library("RColorBrewer")
plotColors = brewer.pal(12, "Paired")

# Data
data <- list(N = 5, y = c(15.77, 20.5, 8.26, 14.37, 21.09))

# Prior
prior <- list(mu0 = 20, kappa0 = 1, nu0 = 5, sigma20 = 5^2)

sm <- stan_model(file = './normalopt.stan')
optres <- optimizing(sm, data = c(data,prior), hessian = TRUE)
optres$par
sqrt(diag(-solve(optres$hessian)))
