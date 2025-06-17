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

  // Likelihood
  x ~ poisson_log(lambda_log);
  y ~ poisson_log(mu_log);
}

generated quantities {
  int x_pred[ngames];
  int y_pred[ngames];
  vector[ngames] log_lik;

  // Generate predictions
  for (k in 1:ngames) {
    x_pred[k] = poisson_log_rng(lambda_log[k]);
    y_pred[k] = poisson_log_rng(mu_log[k]);

    log_lik[k] = poisson_log_lpmf(x[k] | lambda_log[k]) +
                 poisson_log_lpmf(y[k] | mu_log[k]);
  }
}
