data {
  int<lower=1> nteams;       // Número de times
  int<lower=1> ngames;       // Número total de jogos no treino
  int<lower=1> nrounds;      // Número total de rodadas
  array[ngames] int<lower=1, upper=nrounds> i_round; // Rodada de cada jogo
  array[ngames] int<lower=1, upper=nteams> team_name_index; // Índice do time da casa (home)
  array[ngames] int<lower=1, upper=nteams> opponent_index; // Índice do time visitante (away)  
  vector[ngames] gf;           // Gols do time da casa (home goals)
  vector[ngames] ga;           // Gols do time visitante (away goals)
  
  // Dados novos para previsão fora da amostra
  int<lower=1> ngames_new;
  array[ngames_new] int<lower=1, upper=nrounds> i_round_new;
  array[ngames_new] int<lower=1, upper=nteams> team_name_index_new;
  array[ngames_new] int<lower=1, upper=nteams> opponent_index_new;
}

parameters {
  matrix[nrounds, nteams] att_raw;   // habilidade ofensiva (attack) dinâmica
  matrix[nrounds, nteams] def_raw;   // habilidade defensiva (defense) dinâmica

  real home;                     // efeito de jogar em casa
  real<lower=0> sigma_att;       // desvio padrão para variação ataque
  real<lower=0> sigma_def;       // desvio padrão para variação defesa
  real<lower=0> sigma_obs;       // desvio padrão observacional dos gols
}

transformed parameters {
  matrix[nrounds, nteams] att;
  matrix[nrounds, nteams] def;
  vector[ngames] theta1;
  vector[ngames] theta2;

  for (t in 1:nrounds){
    att[t] = att_raw[t] - mean(att_raw[t]);
    def[t] = def_raw[t] - mean(def_raw[t]);
  }

  for (k in 1:ngames) {
    theta1[k] = att[i_round[k], team_name_index[k]] - def[i_round[k], opponent_index[k]] + home;
    theta2[k] = att[i_round[k], opponent_index[k]] - def[i_round[k], team_name_index[k]];
  }
}



model {
  // Priors
  home ~ normal(0, 10);
  sigma_att ~ cauchy(0, 25);
  sigma_def ~ cauchy(0, 25);
  sigma_obs ~ cauchy(0, 25);
  
  target += normal_lpdf(att_raw[1] | 0, sigma_att);
  target += normal_lpdf(def_raw[1] | 0, sigma_def);
  
  // Dinâmica dos parâmetros ataque e defesa ao longo das rodadas	
  for (t in 2:nrounds) {
      target += normal_lpdf(att_raw[t] | att_raw[t-1], sigma_att);
      target += normal_lpdf(def_raw[t] | def_raw[t-1], sigma_def);
  }

  // Likelihood dos gols observados (normal, com média dinâmica)
  for (k in 1:ngames) {
    target += normal_lpdf(gf[k] | theta1[k], sigma_obs);
    target += normal_lpdf(ga[k] | theta2[k], sigma_obs);
  }
}



generated quantities {
  // vector[ngames] log_lik; // desnecessário para a Normal
  vector[ngames_new] theta1_new;
  vector[ngames_new] theta2_new;
  array[ngames_new] int gf_new;
  array[ngames_new] int ga_new;
  
  // Calcula médias para dados novos
  for (k in 1:ngames_new) {
    theta1_new[k] = att[i_round_new[k], team_name_index_new[k]] - def[i_round_new[k], opponent_index_new[k]] + home;
    theta2_new[k] = att[i_round_new[k], opponent_index_new[k]] - def[i_round_new[k], team_name_index_new[k]];
  }
  
  // Amostras preditivas para dados novos
  // gf_new = normal_rng(theta1_new, sigma_obs);
  // ga_new = normal_rng(theta2_new, sigma_obs);
}

