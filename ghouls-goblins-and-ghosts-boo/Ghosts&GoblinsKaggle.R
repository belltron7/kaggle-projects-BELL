# ======================================================================
# Ghouls, Goblins & Ghosts — Tuned Neural Network (nnet MLP)
# Save as: GGG_NN_nnet_tuned.R
# ======================================================================

library(tidyverse)
library(tidymodels)
library(nnet)
library(doParallel)

set.seed(348)

# ----------------- Load Data -----------------
train <- read_csv("train.csv") %>%
  mutate(type = factor(type))

test  <- read_csv("test.csv")

# ----------------- Recipe -----------------
nn_recipe <- recipe(type ~ ., data = train) %>%
  update_role(id, new_role = "id") %>%                # don't use id as predictor
  step_other(all_nominal_predictors(), threshold = 0.01) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_predictors())

# ----------------- CV folds -----------------
folds <- vfold_cv(train, v = 5, strata = type)

# ----------------- Model Specification -----------------
nn_mod <- mlp(
  hidden_units = tune(),   # number of neurons in hidden layer
  penalty      = tune(),   # L2 regularization
  epochs       = 300       # a bit more training than before
) %>%
  set_engine("nnet") %>%
  set_mode("classification")

# ----------------- Workflow -----------------
nn_wf <- workflow() %>%
  add_recipe(nn_recipe) %>%
  add_model(nn_mod)

# ----------------- Tuning Grid -----------------
# hidden_units: 3 to 40
# penalty: 10^-5 to 10^0 (handled on log10 scale internally)
nn_grid <- grid_space_filling(
  hidden_units(range = c(3L, 40L)),
  penalty(range = c(-5, 0)),   # dials::penalty uses log10 scale
  size = 30                    # 30 combinations
)

# ----------------- Parallel Processing (optional but nice) -----------------
cl <- makeCluster(parallel::detectCores() - 1)
registerDoParallel(cl)

# ----------------- Tune -----------------
nn_tune <- tune_grid(
  nn_wf,
  resamples = folds,
  grid = nn_grid,
  metrics = metric_set(accuracy),
  control = control_grid(save_pred = TRUE)
)

stopCluster(cl)

# Quick peek (optional in interactive session)
#print(show_best(nn_tune, metric = "accuracy", n = 10))

# ----------------- Best Params -----------------
best_params <- select_best(nn_tune, metric = "accuracy")

# ----------------- Final Fit on Full Training Data -----------------
nn_final <- finalize_workflow(nn_wf, best_params)

nn_fit <- nn_final %>%
  fit(data = train)

# ----------------- Predict on Test -----------------
preds <- predict(nn_fit, new_data = test, type = "class")

submission <- tibble(
  id   = test$id,
  type = preds$.pred_class
)

write_csv(submission, "GGG_NN_nnet_tuned.csv")

cat("✅ Tuned Neural Net (nnet) complete! Saved as GGG_NN_nnet_tuned.csv\n")
cat("Best hyper-parameters:\n")
print(best_params)
