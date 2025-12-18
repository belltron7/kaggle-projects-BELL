# =============================================================
# WHAT'S COOKING — SPARSE TF-IDF + LINEAR SVM (FINAL VERSION)
# =============================================================

library(tidyverse)
library(tidytext)
library(jsonlite)
library(Matrix)
library(e1071)

set.seed(348)

# ------------------------------
# Load JSON
# ------------------------------
train_raw <- fromJSON("train.json")
test_raw  <- fromJSON("test.json")

train_tidy <- train_raw %>%
  unnest(ingredients) %>%
  rename(ingredient = ingredients)

test_tidy <- test_raw %>%
  unnest(ingredients) %>%
  rename(ingredient = ingredients)

# ------------------------------
# Combine train + test to share vocabulary
# ------------------------------
all_data <- bind_rows(
  train_tidy %>% mutate(cuisine = train_raw$cuisine[match(id, train_raw$id)], split = "train"),
  test_tidy %>% mutate(cuisine = NA, split = "test")
)

# ------------------------------
# Tokenize and count
# ------------------------------
tokens <- all_data %>%
  select(id, ingredient, cuisine, split) %>%
  unnest_tokens(word, ingredient)

# Remove stopwords (optional)
tokens <- tokens %>% anti_join(stop_words, by = "word")

# TF-IDF
word_counts <- tokens %>% count(id, word)
tfidf <- word_counts %>% bind_tf_idf(word, id, n)

# Keep top 4000 informative words
top_words <- tfidf %>%
  group_by(word) %>%
  summarise(avg_tfidf = mean(tf_idf)) %>%
  arrange(desc(avg_tfidf)) %>%
  slice_head(n = 4000) %>%
  pull(word)

tfidf_small <- tfidf %>% filter(word %in% top_words)

# ------------------------------
# Build sparse matrix
# ------------------------------
dtm <- tfidf_small %>% cast_sparse(id, word, tf_idf)

# Split into train/test matrices
train_ids <- train_raw$id
test_ids  <- test_raw$id

x_train <- dtm[as.character(train_ids), ]
x_test  <- dtm[as.character(test_ids), ]
y_train <- factor(train_raw$cuisine)

# ------------------------------
# Train Linear SVM
# ------------------------------
svm_model <- svm(
  x = x_train,
  y = y_train,
  kernel = "linear",
  probability = FALSE
)

# ------------------------------
# Predict
# ------------------------------
preds <- predict(svm_model, x_test)

submission <- tibble(
  id = test_ids,
  cuisine = preds
)

write_csv(submission, "WhatsCooking_SparseTFIDF_SVM.csv")

cat("✅ DONE! Saved 'WhatsCooking_SparseTFIDF_SVM.csv'\n")
