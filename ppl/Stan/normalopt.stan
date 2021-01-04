// Rstan code for the iid normal model from the book 
// Bayesian Learning (https://github.com/mattiasvillani/BayesianLearningBook)

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
    target += normal_lpdf(y | theta, sqrt(sigma2)) 
    + scaled_inv_chi_square_lpdf(sigma2 | nu0, sqrt(sigma20))
    + normal_lpdf(theta | mu0, sqrt(sigma2/kappa0));
}




