############################################################
## Walmart - Improved Random Forest for Kaggle Submission ##
############################################################

## Libraries I need
library(tidyverse)
library(vroom)
library(tidymodels)
library(DataExplorer)
library(lubridate)

## Read in the Data
train    <- vroom("./train.csv")
test     <- vroom("./test.csv")
features <- vroom("./features.csv")
stores   <- vroom("./stores.csv")

#######################################
## Impute Missing Markdowns Features ##
#######################################

features <- features %>%
  mutate(across(starts_with("MarkDown"), ~ replace_na(., 0))) %>%
  mutate(across(starts_with("MarkDown"), ~ pmax(.x, 0))) %>%
  mutate(
    MarkDown_Total = rowSums(across(starts_with("MarkDown")), na.rm = TRUE),
    MarkDown_Flag  = if_else(MarkDown_Total > 0, 1, 0),
    MarkDown_Log   = log1p(MarkDown_Total)
  ) %>%
  select(-MarkDown1, -MarkDown2, -MarkDown3, -MarkDown4, -MarkDown5)

###########################################
## Impute Missing CPI and Unemployment  ##
###########################################

feature_recipe <- recipe(~ ., data = features) %>%
  step_mutate(DecDate = decimal_date(Date)) %>%
  step_impute_bag(CPI, Unemployment,
                  impute_with = imp_vars(DecDate, Store))

imputed_features <- juice(prep(feature_recipe))

########################
## Merge the Datasets ##
########################

fullTrain <- train %>%
  left_join(imputed_features, by = c("Store", "Date")) %>%
  left_join(stores, by = "Store") %>%
  select(-IsHoliday.y) %>%
  rename(IsHoliday = IsHoliday.x)

fullTest <- test %>%
  left_join(imputed_features, by = c("Store", "Date")) %>%
  left_join(stores, by = "Store") %>%
  select(-IsHoliday.y) %>%
  rename(IsHoliday = IsHoliday.x)

## ------------------------------------------------------------
## FIX: Remove Type everywhere to avoid the step_dummy error
## ------------------------------------------------------------
fullTrain <- fullTrain %>% select(-Type)
fullTest  <- fullTest  %>% select(-Type)

##################################
## Loop Through Store–Dept      ##
##################################

all_preds <- tibble(Id = character(), Weekly_Sales = numeric())
n_storeDepts <- fullTest %>% distinct(Store, Dept) %>% nrow()
cntr <- 0

for (store in unique(fullTest$Store)) {
  
  store_train <- fullTrain %>% filter(Store == store)
  store_test  <- fullTest  %>% filter(Store == store)
  
  for (dept in unique(store_test$Dept)) {
    
    dept_train <- store_train %>% filter(Dept == dept)
    dept_test  <- store_test  %>% filter(Dept == dept)
    
    if (nrow(dept_train) == 0) {
      
      preds <- dept_test %>%
        transmute(
          Id = paste(Store, Dept, Date, sep = "_"),
          Weekly_Sales = 0
        )
      
    } else if (nrow(dept_train) < 10) {
      
      preds <- dept_test %>%
        transmute(
          Id = paste(Store, Dept, Date, sep = "_"),
          Weekly_Sales = mean(dept_train$Weekly_Sales)
        )
      
    } else {
      
      my_recipe <- recipe(Weekly_Sales ~ ., data = dept_train) %>%
        step_mutate(Holiday = as.integer(IsHoliday)) %>%
        step_date(Date, features = c("month", "year", "week", "doy", "dow")) %>%
        step_rm(Date, Store, Dept, IsHoliday) %>%
        step_dummy(all_nominal_predictors()) %>%
        step_zv(all_predictors())
      
      prepped_recipe <- prep(my_recipe)
      train_processed <- juice(prepped_recipe)
      
      n_pred <- ncol(train_processed) - 1
      mtry_val <- min(8, max(2, floor(sqrt(n_pred))))
      
      my_model <- rand_forest(
        mtry  = mtry_val,
        trees = 400,
        min_n = 5
      ) %>%
        set_engine("ranger", importance = "impurity") %>%
        set_mode("regression")
      
      my_wf <- workflow() %>%
        add_recipe(my_recipe) %>%
        add_model(my_model) %>%
        fit(dept_train)
      
      preds <- dept_test %>%
        transmute(
          Id = paste(Store, Dept, Date, sep = "_"),
          Weekly_Sales = predict(my_wf, new_data = .) %>% pull(.pred)
        )
    }
    
    all_preds <- bind_rows(all_preds, preds)
    cntr <- cntr + 1
    cat("Store", store, "Dept", dept, "—",
        round(100 * cntr / n_storeDepts, 1), "% complete\n")
  }
}

vroom_write(all_preds, "RandomForestPreds.csv", delim = ",")
