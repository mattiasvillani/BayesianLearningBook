// Rstan code for the iid normal model from the book 
// Bayesian Learning (https://github.com/mattiasvillani/BayesianLearningBook)

// The input data is a vector y of length N.
data {
  int<lower=0> N;
  vector[N] y;
  real<lower=0> sigma;
}

// The parameters in the model
parameters {
  real mu;
}

// The Normal model
//model {
//  y ~ normal(mu, sigma);
//}

model {
  for (n in 1:N)
    y[n] ~ normal(mu, sigma);
}




