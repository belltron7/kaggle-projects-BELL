# =============================================================
#  Facebook Prophet Forecasting Homework
#  For 2 Store–Department combinations
# =============================================================

library(tidyverse)
library(lubridate)
library(vroom)
library(tidymodels)
library(DataExplorer)
library(prophet)

# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------

train    <- vroom("./train.csv")
test     <- vroom("./test.csv")
features <- vroom("./features.csv")

# ------------------------------------------------------------
# 2. Impute MarkDown Variables (professor code)
# ------------------------------------------------------------

features <- features %>%
  mutate(across(starts_with("MarkDown"), ~ replace_na(., 0))) %>%
  mutate(across(starts_with("MarkDown"), ~ pmax(., 0))) %>%
  mutate(
    MarkDown_Total = rowSums(across(starts_with("MarkDown")), na.rm = TRUE),
    MarkDown_Flag  = if_else(MarkDown_Total > 0, 1, 0),
    MarkDown_Log   = log1p(MarkDown_Total)
  ) %>%
  select(-MarkDown1, -MarkDown2, -MarkDown3, -MarkDown4, -MarkDown5)

# ------------------------------------------------------------
# 3. Impute CPI + Unemployment (professor code)
# ------------------------------------------------------------

feature_recipe <- recipe(~., data = features) %>%
  step_mutate(DecDate = decimal_date(Date)) %>%
  step_impute_bag(CPI, Unemployment,
                  impute_with = imp_vars(DecDate, Store))

imputed_features <- juice(prep(feature_recipe))

# ------------------------------------------------------------
# 4. Merge full training + test sets (professor code)
# ------------------------------------------------------------

fullTrain <- left_join(train, imputed_features, 
                       by = c("Store", "Date")) %>%
  select(-IsHoliday.y) %>% 
  rename(IsHoliday = IsHoliday.x)

fullTest <- left_join(test, imputed_features, 
                      by = c("Store", "Date")) %>%
  select(-IsHoliday.y) %>%
  rename(IsHoliday = IsHoliday.x)

# ------------------------------------------------------------
# 5. Choose TWO store–department combos for homework
# ------------------------------------------------------------

sd_list <- tribble(
  ~Store, ~Dept,
  1,      1,
  2,      5       # <-- you can change these if you want
)

results_plots <- list()

# ------------------------------------------------------------
# 6. LOOP through both store-dept combos
# ------------------------------------------------------------

for(i in 1:nrow(sd_list)){
  
  store_i <- sd_list$Store[i]
  dept_i  <- sd_list$Dept[i]
  
  message("\n-------------------------------------")
  message("Running Prophet for Store ", store_i, " Dept ", dept_i)
  message("-------------------------------------\n")
  
  # Filter training + test
  sd_train <- fullTrain %>%
    filter(Store == store_i, Dept == dept_i) %>%
    rename(y = Weekly_Sales, ds = Date)
  
  sd_test <- fullTest %>%
    filter(Store == store_i, Dept == dept_i) %>%
    rename(ds = Date)
  
  # If no training data → skip this one
  if(nrow(sd_train) == 0){
    warning("NO TRAINING DATA FOR STORE ", store_i, " DEPT ", dept_i)
    next
  }
  
  # ------------------------------------------------------------
  # 7. Build Prophet model with regressors
  # ------------------------------------------------------------
  
  prophet_model <- prophet()
  
  # Add regressors if they exist in dataset
  regressors <- c("Temperature", "Fuel_Price", "CPI",
                  "Unemployment", "MarkDown_Log", "MarkDown_Flag")
  
  for(r in regressors){
    if(r %in% names(sd_train)){
      prophet_model <- add_regressor(prophet_model, r)
    }
  }
  
  # Fit model
  prophet_model <- fit.prophet(prophet_model, sd_train)
  
  # ------------------------------------------------------------
  # 8. Predict (both fitted & forecast)
  # ------------------------------------------------------------
  
  fitted_vals <- predict(prophet_model, sd_train)
  test_preds  <- predict(prophet_model, sd_test)
  
  # ------------------------------------------------------------
  # 9. Store plot
  # ------------------------------------------------------------
  
  p <- ggplot() +
    geom_line(data = sd_train,
              aes(x = ds, y = y, color = "Actual")) +
    geom_line(data = fitted_vals,
              aes(x = as.Date(ds), y = yhat, color = "Fitted")) +
    geom_line(data = test_preds,
              aes(x = as.Date(ds), y = yhat, color = "Forecast")) +
    scale_color_manual(values = c(
      "Actual"   = "black",
      "Fitted"   = "blue",
      "Forecast" = "red"
    )) +
    labs(
      title = paste0("Store ", store_i, " Dept ", dept_i,
                     " — Prophet Forecast"),
      x = "Date", y = "Weekly Sales", color = ""
    ) +
    theme_minimal(base_size = 14)
  
  results_plots[[i]] <- p
}

# ------------------------------------------------------------
# 10. Combine plots side-by-side
# ------------------------------------------------------------
library(patchwork)

final_plot <- results_plots[[1]] + results_plots[[2]]

print(final_plot)

# ------------------------------------------------------------
# 11. Save output for homework submission
# ------------------------------------------------------------

ggsave("prophet_homework_2panel.png", final_plot,
       width = 14, height = 6, dpi = 320)
