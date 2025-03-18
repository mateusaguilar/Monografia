# Pacotes Necessários
#install.packages("glmmTMB")
library(glmmTMB)
library(tidyverse)

#devtools::install_github("salvnetto/DYNMO")

#remotes::install_github("salvnetto/DYNMO")

library(FootStats)
matches = load_data("brazil", "brasileirao_a", "match_history") #Historico de partidas
players = load_data("brazil", "brasileirao_a", "squad") #Jogadores

View(matches)

matches$resultado <- factor(ifelse(matches$gf > matches$ga, 2,
                            ifelse(matches$gf == matches$ga, 1, 0)))

matches$home = 0
matches$home[which(matches$venue == 'Home')] = 1

matches$def = -1
matches$atk = 1

# Separando bases
df_train <- subset(matches, season == 2024 & round <= 34)
df_test <- subset(matches, season == 2024 & round > 34)

# Modelos Poisson
fit1 <- glmmTMB(gf ~ 1 + (0 + atk + home | team_name) + (0 + def | opponent),
                data = df_train, 
                family = poisson(link = 'log'))
summary(fit1)
a = ranef(fit1)

atk = data.frame(atk = a$cond$team_name$atk, team_name = row.names(a$cond$team_name))
def = data.frame(a$cond$opponent, opponent = row.names(a$cond$opponent))
home = data.frame(home = a$cond$team_name$home, team_name = row.names(a$cond$team_name))

c = coef(fit1)
d = c$cond$team_name[1, c("(Intercept)")]
vars <- data.frame(model.matrix(fit1)) 
vars2 <- data.frame('Intercept' = vars[,1] * d[1])


table <- df_train %>% 
  select(gf, ga, round, venue, team_name, opponent) %>% 
  bind_cols(vars2) %>% 
  bind_cols(data.frame(Predito = predict(fit1))) %>% 
  merge(atk, by = 'team_name') %>% 
  merge(def, by = 'opponent') %>% 
  merge(home, by = 'team_name')

View(table)


df_test <- df_test %>% 
  select(gf, ga, round, venue, team_name, opponent, home, atk, def, resultado)

# Previsões para gols marcados (gf_hat)
df_test$gf_hat <- predict(fit1, df_test, type = "response")

# Criar um novo dataframe para prever gols sofridos (invertendo times)
df_test_ga <- df_test
df_test_ga$team_name <- df_test$opponent
df_test_ga$opponent <- df_test$team_name
df_test_ga <- df_test_ga %>% rename(ga_hat = 'gf_hat')

#df_test$ga_hat <- df_test_ga$gf_hat

df_test_ga <- df_test_ga %>% 
           select(team_name, opponent, ga_hat)
df_test <- merge(df_test, df_test_ga, by = c('team_name', 'opponent'))

# Previsão para gols sofridos (ga_hat)
#df_test$ga_hat <- predict(fit1, df_test_ga, type = "response")

df_test2 <- df_test %>% subset(venue == 'Home')

# Classificando vitória (2), empate (1) ou derrota (0)
df_test2$pred_result <- factor(ifelse(round(df_test2$gf_hat) > round(df_test2$ga_hat), 2,
                                     ifelse(round(df_test2$gf_hat) == round(df_test2$ga_hat), 1, 0)))

# Matriz de Confusão
conf_matrix <- table(Predito = df_test2$pred_result, Real = df_test2$resultado)
print(conf_matrix)

# Acurácia geral
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
print(paste("Acurácia: ", round(accuracy, 3)))

