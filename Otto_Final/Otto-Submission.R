library(tidyverse)
library(tidymodels)
library(xgboost)
library(vip)
library(glmnet)
library(janitor)
library(corrplot)
library(ranger)
library(data.table)
train_path <- "/home/peely/STAT348/Otto_Final/otto-group-product-classification-challenge/train.csv"
test_path  <- "/home/peely/STAT348/Otto_Final/otto-group-product-classification-challenge/test.csv"
sample_path <- "/home/peely/STAT348/Otto_Final/otto-group-product-classification-challenge/sampleSubmission.csv"

train <- read_csv(train_path)
test  <- read_csv(test_path)
sample_submission <- read_csv(sample_path)

glimpse(train)
glimpse(test)

train <- train %>%
  mutate(target = factor(target))

levels(train$target)

train %>%
  count(target) %>%
  mutate(prop = n / sum(n))

nzv <- caret::nearZeroVar(train %>% select(-id, -target))
length(nzv)

summary(train %>% select(-id))

corrplot::corrplot(
  cor(train %>% select(-id, -target) %>% sample_n(2000)),
  method = "color",
  tl.cex = 0.5
)

train_x <- train %>% select(-id, -target)
train_y <- train$target

test_x <- test %>% select(-id)


library(tidymodels)
set.seed(123)

# Remove NZV columns from training AND test
nzv_cols <- caret::nearZeroVar(train_x)
train_x_clean <- train_x %>% select(-all_of(nzv_cols))
test_x_clean  <- test_x %>% select(-all_of(nzv_cols))

# Combine back with target
train_clean <- train_x_clean %>%
  mutate(target = train_y)

# Split training data
set.seed(123)
otto_split <- initial_split(train_clean, prop = 0.8, strata = target)
otto_train <- training(otto_split)
otto_test  <- testing(otto_split)

# 5-fold stratified CV
otto_folds <- vfold_cv(otto_train, v = 5, strata = target)

# Define recipe: keep it simple
otto_recipe <- recipe(target ~ ., data = otto_train) %>%
  step_normalize(all_predictors(), -all_nominal()) %>%   # not required for trees, but helps xgboost
  step_zv(all_predictors())



library(ranger)
library(tidymodels)

# Model specification (no tuning yet)
rf_model <- rand_forest(
  mtry  = 20,
  min_n = 5,
  trees = 1000
) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

# Workflow
rf_wf <- workflow() %>%
  add_model(rf_model) %>%
  add_recipe(otto_recipe)

# Fit using cross-validation
set.seed(123)
rf_res <- fit_resamples(
  rf_wf,
  otto_folds,
  metrics = metric_set(accuracy, mn_log_loss),
  control = control_resamples(save_pred = TRUE)
)

collect_metrics(rf_res)


library(xgboost)
library(tidymodels)

xgb_model <- boost_tree(
  trees = 1000,
  learn_rate = 0.05,
  tree_depth = 6,
  mtry = 20,       # similar to RF suggestion
  min_n = 5,       # same as teammate's setting
  loss_reduction = 0,
  sample_size = 1
) %>%
  set_engine("xgboost", nthread = parallel::detectCores()) %>%
  set_mode("classification")

xgb_wf <- workflow() %>%
  add_recipe(otto_recipe) %>%
  add_model(xgb_model)

set.seed(123)
xgb_res <- fit_resamples(
  xgb_wf,
  otto_folds,
  metrics = metric_set(accuracy, mn_log_loss),
  control = control_resamples(save_pred = TRUE)
)

collect_metrics(xgb_res)


# Fit final model on ALL training data
final_xgb_fit <- fit(
  xgb_wf,
  data = train_clean
)

# Predict probabilities for test set
xgb_preds <- predict(
  final_xgb_fit,
  new_data = test_x_clean,
  type = "prob"
)

# Add id column back in
submission <- bind_cols(
  tibble(id = test$id),
  xgb_preds
)

# Rename columns to match Kaggle format
names(submission) <- c("id", paste0("Class_", 1:9))

# Write to CSV
write_csv(submission, "xgb_submission.csv")
