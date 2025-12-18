# ==========================================================
# AMAZON — KNN CLASSIFIER
# ==========================================================

library(tidymodels)


setwd("/home/peely/STAT348/AmazonEmployeeAccess")

sample <- read_csv("sampleSubmission.csv")
test   <- read_csv("test.csv")
train  <- read_csv("train.csv")

train <- train %>% mutate(ACTION = factor(ACTION))

# ------- RECIPE (dummy + normalize) -------
knn_recipe <- recipe(ACTION ~ ., data = train) %>%
  step_mutate_at(all_predictors(), fn = as.factor) %>%
  step_other(all_nominal_predictors(), threshold = 0.001) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

# ------- MODEL SPEC -------
knn_model <- nearest_neighbor(
  neighbors = tune(),
  weight_func = "rectangular"
) %>%
  set_engine("kknn") %>%
  set_mode("classification")

knn_wf <- workflow() %>%
  add_recipe(knn_recipe) %>%
  add_model(knn_model)

# ------- TUNING -------
set.seed(348)
folds <- vfold_cv(train, v = 5)

knn_grid <- tibble(neighbors = seq(3, 75, by = 4))

knn_tune <- tune_grid(
  knn_wf,
  resamples = folds,
  grid = knn_grid,
  metrics = metric_set(roc_auc)
)

best_k <- select_best(knn_tune, "roc_auc")

# ------- FINAL FIT -------
final_knn <- finalize_workflow(knn_wf, best_k) %>%
  fit(data = train)

# ------- PREDICTIONS TO KAGGLE -------
knn_preds <- predict(final_knn, new_data = test, type = "prob") %>%
  select(.pred_1) %>%
  rename(ACTION = .pred_1)

kaggle_knn <- bind_cols(test %>% select(id), knn_preds)

write_csv(kaggle_knn, "KNNpreds.csv")

print("✅ KNN complete — wrote KNNpreds.csv")
print(best_k)
# ==========================================================
