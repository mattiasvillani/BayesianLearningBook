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

//transformed parameters {
  // real<lower=0> sigma = sqrt(sigma2);
  // real<lower=0> cv = sigma/theta;   
//}

model {
    sigma2 ~ scaled_inv_chi_square(nu0, sqrt(sigma20));
    theta ~ normal(mu0,sqrt(sigma2/kappa0));
    y ~ normal(theta, sqrt(sigma2));
}




