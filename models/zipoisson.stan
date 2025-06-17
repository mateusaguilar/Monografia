data {
  int<lower=1> ngames;
  int<lower=1> nteams;
  int<lower=1, upper=nteams> i[ngames]; // Home team index
  int<lower=1, upper=nteams> j[ngames]; // Away team index
  int<lower=0> x[ngames];
  int<lower=0> y[ngames];
}

parameters {
  vector[nteams] home_raw;
  vector[nteams] att_raw;
  vector[nteams] def_raw;
  real<lower=0, upper=1> pzero_x;
  real<lower=0, upper=1> pzero_y;
}

transformed parameters {
  vector[nteams] home;
  vector[nteams] att;
  vector[nteams] def;
  vector[ngames] lambda_log; // Linear predictor Home
  vector[ngames] mu_log;     // Linear predictor Away

  // Constrained parameters (ensure sum(att) = 0, sum(def) = 0 and sum(home) = 0)
  home = home_raw - mean(home_raw);
  att = att_raw - mean(att_raw);
  def = def_raw - mean(def_raw);

  lambda_log = att[i] - def[j] + home[i];
  mu_log = att[j] - def[i];
}

model {
  // Priors
  home_raw ~ normal(0, 10);
  att_raw ~ normal(0, 10);
  def_raw ~ normal(0, 10);
  pzero_x ~ beta(1, 1);
  pzero_y ~ beta(1, 1);

  // Likelihood
  for (k in 1:ngames){
    // HOME
    if (x[k] == 0)
      target += log_sum_exp(
        bernoulli_lpmf(1 | pzero_x),
        bernoulli_lpmf(0 | pzero_x) + poisson_log_lpmf(x[k] | lambda_log[k])
        );
    else
      target += bernoulli_lpmf(0 | pzero_x) + poisson_log_lpmf(x[k] | lambda_log[k]);

    // AWAY
    if (y[k] == 0)
      target += log_sum_exp(
        bernoulli_lpmf(1 | pzero_y),
        bernoulli_lpmf(0 | pzero_y) + poisson_log_lpmf(y[k] | mu_log[k])
        );
    else
      target += bernoulli_lpmf(0 | pzero_y) + poisson_log_lpmf(x[k] | mu_log[k]);
  }
}

generated quantities {
  int x_pred[ngames];
  int y_pred[ngames];
  vector[ngames] log_lik;

  // Generate predictions
  for (k in 1:ngames){
    // HOME PREDICTIONS
    x_pred[k] = poisson_log_rng(lambda_log[k]);
    if (x_pred[k] == 0)
      x_pred[k] = bernoulli_rng(pzero_x) == 1 ? 0 : poisson_log_rng(lambda_log[k]);

    // AWAY PREDICTIONS
    y_pred[k] = poisson_log_rng(mu_log[k]);
    if (y_pred[k] == 0)
      y_pred[k] = bernoulli_rng(pzero_y) == 1 ? 0 : poisson_log_rng(mu_log[k]);


    log_lik[k] = 0;
    // HOME
    if (x[k] == 0)
      log_lik[k] += log_sum_exp(
        bernoulli_lpmf(1 | pzero_x),
        bernoulli_lpmf(0 | pzero_x) + poisson_log_lpmf(x[k] | lambda_log[k])
        );
    else
      log_lik[k] += bernoulli_lpmf(0 | pzero_x) + poisson_log_lpmf(x[k] | lambda_log[k]);

    // AWAY
    if (y[k] == 0)
      log_lik[k] += log_sum_exp(
        bernoulli_lpmf(1 | pzero_y),
        bernoulli_lpmf(0 | pzero_y) + poisson_log_lpmf(y[k] | mu_log[k])
        );
    else
      log_lik[k] += bernoulli_lpmf(0 | pzero_y) + poisson_log_lpmf(x[k] | mu_log[k]);
  }
}
