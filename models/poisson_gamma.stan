data {
  int<lower=1> ngames;
  int<lower=1> nteams;
  int<lower=1, upper=nteams> i[ngames]; // Home team index
  int<lower=1, upper=nteams> j[ngames]; // Away team index
  int<lower=0> x[ngames];
  int<lower=0> y[ngames];
}

parameters {
  vector<lower=0>[nteams] home;
  vector<lower=0>[nteams] att;
  vector<lower=0>[nteams] def;
}

transformed parameters {
  vector[ngames] lambda_log; // Linear predictor Home
  vector[ngames] mu_log;     // Linear predictor Away

  lambda_log = att[i] - def[j] + home[i];
  mu_log = att[j] - def[i];
}

model {
  // Priors
  home ~ gamma(0.1, 0.1);
  att ~ gamma(0.1, 0.1);
  def ~ gamma(0.1, 0.1);

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
