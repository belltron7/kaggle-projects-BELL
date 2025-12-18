# ===================================================================
# 🚲 BIKE SHARE FINAL PIPELINE — TARGET: < 0.40 RMSE
# Includes: Linear | Penalized | Tree | Boosted | Stacked (H2O)
# ===================================================================

# --- SETUP ---
setwd("/home/peely/STAT348/KaggleBikeShare")

library(tidyverse)
library(tidymodels)
library(vroom)
library(lubridate)
library(patchwork)
library(rpart)
library(lightgbm)
library(bonsai)
library(h2o)
library(agua)

# --- READ DATA ---
train <- vroom("train.csv") %>%
  mutate(
    datetime = as.POSIXct(datetime),
    hour = lubridate::hour(datetime),
    date = lubridate::date(datetime),
    wday = lubridate::wday(datetime),
    log_count = log1p(count)
  ) %>%
  select(-casual, -registered, -count)

test <- vroom("test.csv") %>%
  mutate(
    datetime = as.POSIXct(datetime),
    hour = lubridate::hour(datetime),
    date = lubridate::date(datetime),
    wday = lubridate::wday(datetime)
  )

# --- FEATURE ENGINEERING ---
mybike_recipe <- recipe(log_count ~ ., data = train) %>%
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>%
  step_mutate(weather = factor(weather)) %>%
  step_mutate(season = factor(season)) %>%
  step_time(datetime, features = "hour") %>%
  step_date(datetime, features = c("month", "year")) %>%
  step_mutate(is_weekend = ifelse(wday %in% c(1, 7), 1, 0)) %>%
  step_interact(terms = ~ hour:workingday) %>%
  step_ns(temp, deg_free = 4) %>% # nonlinear temperature
  step_dummy(all_nominal_predictors()) %>%
  step_rm(datetime, date) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_corr(all_numeric_predictors(), threshold = 0.7) %>%
  step_zv(all_predictors())

prepped_recipe <- prep(mybike_recipe)
baked_data <- bake(prepped_recipe, new_data = train)
head(baked_data, 5)

# --- BASELINE LINEAR REGRESSION ---
lm_model <- linear_reg() %>% set_engine("lm") %>% set_mode("regression")

lm_workflow <- workflow() %>%
  add_recipe(mybike_recipe) %>%
  add_model(lm_model)

lm_fit <- fit(lm_workflow, data = train)

bike_predictions <- predict(lm_fit, new_data = test) %>%
  mutate(count = exp(.pred) - 1, count = pmax(0, count))

kaggle_submission <- test %>%
  select(datetime) %>%
  bind_cols(bike_predictions %>% select(count)) %>%
  mutate(datetime = as.character(format(datetime)))

vroom_write(kaggle_submission, file = "LinearPreds.csv", delim = ",")

# --- TUNED PENALIZED REGRESSION (GLMNET) ---
set.seed(348)
preg_model <- linear_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet") %>% set_mode("regression")

preg_wf <- workflow() %>% add_recipe(mybike_recipe) %>% add_model(preg_model)
folds <- vfold_cv(train, v = 5)
grid <- grid_regular(penalty(), mixture(), levels = c(penalty = 20, mixture = 5))

cv_res <- tune_grid(preg_wf, resamples = folds, grid = grid, metrics = metric_set(rmse, mae))
best <- select_best(cv_res, metric = "rmse")

final_wf <- finalize_workflow(preg_wf, best) %>% fit(data = train)

tuned_preds <- predict(final_wf, new_data = test) %>%
  mutate(count = exp(.pred) - 1, count = pmax(0, count))

kaggle_tuned <- test %>%
  select(datetime) %>%
  bind_cols(tuned_preds %>% select(count)) %>%
  mutate(datetime = as.character(format(datetime)))

vroom_write(kaggle_tuned, "TunedGlmnet.csv", delim = ",")

# --- REGRESSION TREE MODEL ---
tree_model <- decision_tree(tree_depth = tune(), cost_complexity = tune(), min_n = tune()) %>%
  set_engine("rpart") %>% set_mode("regression")

tree_wf <- workflow() %>% add_recipe(mybike_recipe) %>% add_model(tree_model)

set.seed(348)
tree_folds <- vfold_cv(train, v = 5)
tree_grid <- grid_regular(tree_depth(range = c(2, 12)),
                          min_n(range = c(2, 30)),
                          cost_complexity(range = c(-5, -1)), levels = 5)

tree_res <- tune_grid(tree_wf, resamples = tree_folds, grid = tree_grid,
                      metrics = metric_set(rmse, mae))
best_tree <- select_best(tree_res, metric = "rmse")

final_tree <- finalize_workflow(tree_wf, best_tree) %>% fit(data = train)

tree_preds <- predict(final_tree, new_data = test) %>%
  mutate(count = exp(.pred) - 1, count = pmax(0, count))

kaggle_tree <- test %>% select(datetime) %>%
  bind_cols(tree_preds %>% select(count)) %>%
  mutate(datetime = as.character(format(datetime)))

vroom_write(kaggle_tree, "TreePreds.csv", delim = ",")

# --- BOOSTED TREE MODEL (LightGBM) ---
boost_model <- boost_tree(tree_depth = tune(), trees = tune(), learn_rate = tune()) %>%
  set_engine("lightgbm") %>% set_mode("regression")

boost_wf <- workflow() %>% add_recipe(mybike_recipe) %>% add_model(boost_model)
boost_folds <- vfold_cv(train, v = 5)

boost_grid <- grid_regular(
  tree_depth(range = c(3, 12)),
  trees(range = c(500, 2500)),
  learn_rate(range = c(-3.5, -1.3)),
  levels = 6
)

boost_res <- tune_grid(boost_wf, resamples = boost_folds, grid = boost_grid,
                       metrics = metric_set(rmse, mae))

best_boost <- select_best(boost_res, metric = "rmse")

final_boost <- finalize_workflow(boost_wf, best_boost) %>% fit(data = train)

boost_preds <- predict(final_boost, new_data = test) %>%
  mutate(count = exp(.pred) - 1, count = pmax(0, count))

kaggle_boost <- test %>% select(datetime) %>%
  bind_cols(boost_preds %>% select(count)) %>%
  mutate(datetime = as.character(format(datetime)))

vroom_write(kaggle_boost, "BoostedPreds.csv", delim = ",")

# --- STACKED MODEL (H2O AutoML) ---
h2o::h2o.init(strict_version_check = FALSE)

stack_model <- auto_ml() %>%
  set_engine("h2o",
             max_runtime_secs = 360,  # 6 minutes
             max_models = 15) %>%
  set_mode("regression")

stack_wf <- workflow() %>% add_recipe(mybike_recipe) %>% add_model(stack_model)

final_stack <- fit(stack_wf, data = train)

stack_preds <- predict(final_stack, new_data = test) %>%
  mutate(count = exp(.pred) - 1, count = pmax(0, count))

stack_submission <- test %>% select(datetime) %>%
  bind_cols(stack_preds %>% select(count)) %>%
  mutate(datetime = as.character(format(datetime)))

vroom_write(stack_submission, "StackedPreds.csv", delim = ",")

# --- CLEAN SHUTDOWN ---
h2o::h2o.shutdown(prompt = FALSE)

# ✅ DONE — Kaggle-ready submissions:
# LinearPreds.csv
# TunedGlmnet.csv
# TreePreds.csv
# BoostedPreds.csv
# StackedPreds.csv
# ===================================================================
