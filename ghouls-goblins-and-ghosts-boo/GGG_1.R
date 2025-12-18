# ======================================================================
# Ghouls, Goblins & Ghosts — Baseline Random Forest
# ======================================================================


library(tidyverse)
library(tidymodels)

# ----------------- Load Data -----------------
train <- read_csv("train.csv")
test  <- read_csv("test.csv")

train <- train %>% mutate(type = factor(type))  # target must be factor

# ----------------- Recipe -----------------
rf_recipe <- recipe(type ~ ., data = train) %>%
  update_role(id, new_role = "id") %>%
  step_other(all_nominal_predictors(), threshold = 0.01) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_predictors())

# ----------------- Model -----------------
rf_mod <- rand_forest(
  mtry = 3,
  trees = 500,
  min_n = 5
) %>%
  set_mode("classification") %>%
  set_engine("ranger")

rf_wf <- workflow() %>% add_model(rf_mod) %>% add_recipe(rf_recipe)

# ----------------- Fit -----------------
set.seed(348)
rf_fit <- rf_wf %>% fit(data = train)

# ----------------- Predict -----------------
preds <- predict(rf_fit, new_data = test) %>%
  rename(type = .pred_class)  # rename to match Kaggle format

# ----------------- Save Submission -----------------
submission <- bind_cols(test %>% select(id), preds)
write_csv(submission, "GGG_baseline_RF.csv")

cat("✅ Ghouls & Goblins baseline RF complete\n")

