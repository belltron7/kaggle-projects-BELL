# =====================================================================
# WHAT’S COOKING — Tidy Text + Feature Engineering + ML Baseline
# =====================================================================

library(tidyverse)
library(tidytext)
library(jsonlite)
library(tidymodels)

set.seed(348)

# ------------------------------
# Load JSON Data
# ------------------------------

train_raw <- fromJSON("train.json")
test_raw  <- fromJSON("test.json")

glimpse(train_raw)

# ------------------------------
# Convert to Tidy Text Structure
# ------------------------------
train_tidy <- train_raw %>%
  unnest(ingredients) %>%
  rename(ingredient = ingredients)

test_tidy <- test_raw %>%
  unnest(ingredients) %>%
  rename(ingredient = ingredients)

# ------------------------------
# FEATURE 1 — Ingredient Count
# ------------------------------
feat_count <- train_tidy %>%
  count(id, name = "num_ingredients")

test_feat_count <- test_tidy %>%
  count(id, name = "num_ingredients")

# ------------------------------
# FEATURE 2 — Rare Ingredient Count
# ------------------------------
ingredient_freq <- train_tidy %>%
  count(ingredient, sort = TRUE)

rare_words <- ingredient_freq %>%
  filter(n < 15) %>%
  pull(ingredient)

feat_rare <- train_tidy %>%
  mutate(rare = ingredient %in% rare_words) %>%
  group_by(id) %>%
  summarise(num_rare = sum(rare))

test_feat_rare <- test_tidy %>%
  mutate(rare = ingredient %in% rare_words) %>%
  group_by(id) %>%
  summarise(num_rare = sum(rare))

# ------------------------------
# FEATURE 3 — Common Ingredient Count
# ------------------------------
common_list <- c("garlic", "sugar", "rice", "tomato", "cilantro", "soy sauce")

feat_common <- train_tidy %>%
  mutate(common = ingredient %in% common_list) %>%
  group_by(id) %>%
  summarise(num_common = sum(common))

test_feat_common <- test_tidy %>%
  mutate(common = ingredient %in% common_list) %>%
  group_by(id) %>%
  summarise(num_common = sum(common))

# ------------------------------
# Combine Features
# ------------------------------
train_features <- train_raw %>%
  select(id, cuisine) %>%
  left_join(feat_count, by = "id") %>%
  left_join(feat_rare, by = "id") %>%
  left_join(feat_common, by = "id") %>%
  mutate(cuisine = factor(cuisine))

test_features <- test_raw %>%
  select(id) %>%
  left_join(test_feat_count, by = "id") %>%
  left_join(test_feat_rare, by = "id") %>%
  left_join(test_feat_common, by = "id")

# ------------------------------
# Baseline Random Forest Model
# ------------------------------
recipe_model <- recipe(cuisine ~ ., data = train_features) %>%
  update_role(id, new_role = "id") %>%
  step_normalize(all_numeric_predictors())

rf_mod <- rand_forest(
  trees = 500,
  mtry = 3,
  min_n = 5
) %>%
  set_mode("classification") %>%
  set_engine("ranger")

wf <- workflow() %>%
  add_recipe(recipe_model) %>%
  add_model(rf_mod)

rf_fit <- wf %>% fit(train_features)

# ------------------------------
# Predict + Save Submission
# ------------------------------
preds <- predict(rf_fit, test_features) %>%
  bind_cols(test_features %>% select(id))

write_csv(preds %>% rename(cuisine = .pred_class),
          "WhatsCooking_RF_submission.csv")

cat("✅ Submission saved as WhatsCooking_RF_submission.csv\n")