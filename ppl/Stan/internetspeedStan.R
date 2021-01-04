# Using Rstan to fit the iid normal model on the internet speed data from the 
# book 'Bayesian Learning' by Mattias Villani (https://mattiasvillani.com)
# Book: https://github.com/mattiasvillani/BayesianLearningBook)

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

# Sample from posterior using Stan (the file normal.stan defines the model)
fit <- stan(file = './normal.stan', data = c(data,prior), iter = 10000 )

# print and plot using stan's standard functions
print(fit, pars = c("theta","sigma2"), probs=c(.1,.5,.9))
plot(fit, pars = c("theta", "sigma2"))
pairs(fit, pars = c("theta", "sigma2"), las = 1)
traceplot(fit, pars = c("theta", "sigma2"), inc_warmup = FALSE, nrow = 2)

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

