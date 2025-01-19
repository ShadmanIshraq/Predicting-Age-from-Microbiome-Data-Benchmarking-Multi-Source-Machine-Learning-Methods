library(caret)
library(randomForest)
library(xgboost)
library(gbm)
library(ggplot2)
library(IntegratedLearner)

options(java.parameters = "-Xmx5g") # This is needed for running BART


library(tidyverse) 
library(SuperLearner)
library(cowplot)
library(mcmcplots)
library(bartMachine)






############ Models trained on single-omic dataset



# Set seed for reproducibility
set.seed(123)


# Split data into train and test sets
train_indices_rf <- createDataPartition(filtered_df$age, p = 0.8, list = FALSE)
train_df_rf <- filtered_df[train_indices_rf, ]
test_df_rf <- filtered_df[-train_indices_rf, ]

# Specify train control for 10-fold cross-validation
train_control_rf <- trainControl(method = "cv", 
                                 number = 10, 
                                 savePredictions = "final",
                                 allowParallel = TRUE, 
                                 verboseIter = TRUE)

# Specify hyperparameter tuning grid
tune_grid_rf <- expand.grid(mtry = c(7,35,53, 64,79, 91))

# Train the model on the training set
rf_model <- train(age ~ ., 
                  data = train_df_rf, 
                  method = "rf",
                  trControl = train_control_rf,
                  tuneGrid = tune_grid_rf,
                  ntree = 400,
                  metric = "MAE")


# Set up single round of cross-validation
repeats <- 1
folds <- 10

# Perform cross-validation on the test set
fold_list_rf <- createFolds(test_df_rf$age, k = folds, list = TRUE)

# Initialize vectors to store MAE, R² values, actual, and predicted values
mae_values_rf <- numeric()
rsquared_values_rf <- numeric()
actual_values_rf <- numeric()
predicted_values_rf <- numeric()

# Perform cross-validation and collect MAE and R² for each fold
for (fold in 1:folds) {
  test_indices_rf <- fold_list_rf[[fold]]
  test_fold_rf <- test_df_rf[test_indices_rf, ]
  
  # Predict for the current fold
  predictions_rf <- predict(rf_model, newdata = test_fold_rf[, -which(names(test_fold_rf) == "age")])
  
  # Calculate MAE for the current fold
  mae_fold_rf <- mean(abs(predictions_rf - test_fold_rf$age))
  mae_values_rf <- c(mae_values_rf, mae_fold_rf)
  
  # Calculate R² for the current fold
  ss_total <- sum((test_fold_rf$age - mean(test_fold_rf$age))^2)
  ss_residual <- sum((test_fold_rf$age - predictions_rf)^2)
  rsquared_fold_rf <- 1 - (ss_residual / ss_total)
  rsquared_values_rf <- c(rsquared_values_rf, rsquared_fold_rf)
  
  # Store actual and predicted values for the jitter plot
  actual_values_rf <- c(actual_values_rf, test_fold_rf$age)
  predicted_values_rf <- c(predicted_values_rf, predictions_rf)
}

# Calculate the mean MAE and R² across all folds
mean_mae_rf <- mean(mae_values_rf)
mean_rsquared_rf <- mean(rsquared_values_rf)

# Print the mean MAE and R² on the test set
print(paste("Mean MAE on test set:", mean_mae_rf))
print(paste("Mean R-squared on test set:", mean_rsquared_rf))














# Set seed for reproducibility
set.seed(123)


# Split data into train and test sets
train_indices_xgb <- createDataPartition(filtered_df$age, p = 0.8, list = FALSE)
train_df_xgb <- filtered_df[train_indices_xgb, ]
test_df_xgb <- filtered_df[-train_indices_xgb, ]

# Specify train control for 10-fold cross-validation
train_control_xgb <- trainControl(method = "cv", 
                                  number = 10, 
                                  savePredictions = "final",
                                  allowParallel = TRUE, 
                                  verboseIter = TRUE)

# Specify hyperparameter tuning grid for XGBoost
tune_grid_xgb <- expand.grid(nrounds = c(100, 200),
                             max_depth = c(3, 6, 9),
                             eta = c(0.01, 0.1, 0.3),
                             gamma = c(0, 0.1, 0.3),
                             colsample_bytree = 0.8,
                             min_child_weight = 1,
                             subsample = 0.8)

# Train the model on the training set
xgb_model <- train(age ~ ., 
                   data = train_df_xgb, 
                   method = "xgbTree",
                   trControl = train_control_xgb,
                   tuneGrid = tune_grid_xgb,
                   metric = "MAE")

# Set up single round of cross-validation
repeats <- 1
folds <- 10

# Perform cross-validation on the test set
fold_list_xgb <- createFolds(test_df_xgb$age, k = folds, list = TRUE)

# Initialize vectors to store MAE, R² values, actual, and predicted values
mae_values_xgb <- numeric()
rsquared_values_xgb <- numeric()
actual_values_xgb <- numeric()
predicted_values_xgb <- numeric()

# Perform cross-validation and collect MAE and R² for each fold
for (fold in 1:folds) {
  test_indices_xgb <- fold_list_xgb[[fold]]
  test_fold_xgb <- test_df_xgb[test_indices_xgb, ]
  
  # Predict for the current fold
  predictions_xgb <- predict(xgb_model, newdata = test_fold_xgb[, -which(names(test_fold_xgb) == "age")])
  
  # Calculate MAE for the current fold
  mae_fold_xgb <- mean(abs(predictions_xgb - test_fold_xgb$age))
  mae_values_xgb <- c(mae_values_xgb, mae_fold_xgb)
  
  # Calculate R² for the current fold
  ss_total <- sum((test_fold_xgb$age - mean(test_fold_xgb$age))^2)
  ss_residual <- sum((test_fold_xgb$age - predictions_xgb)^2)
  rsquared_fold_xgb <- 1 - (ss_residual / ss_total)
  rsquared_values_xgb <- c(rsquared_values_xgb, rsquared_fold_xgb)
  
  # Store actual and predicted values for the jitter plot
  actual_values_xgb <- c(actual_values_xgb, test_fold_xgb$age)
  predicted_values_xgb <- c(predicted_values_xgb, predictions_xgb)
}

# Calculate the mean MAE and R² across all folds
mean_mae_xgb <- mean(mae_values_xgb)
mean_rsquared_xgb <- mean(rsquared_values_xgb)

# Print the mean MAE and R² on the test set
print(paste("Mean MAE on test set:", mean_mae_xgb))
print(paste("Mean R-squared on test set:", mean_rsquared_xgb))











# Set seed for reproducibility
set.seed(123)


# Split data into train and test sets
train_indices_gbm <- createDataPartition(filtered_df$age, p = 0.8, list = FALSE)
train_df_gbm <- filtered_df[train_indices_gbm, ]
test_df_gbm <- filtered_df[-train_indices_gbm, ]

# Specify train control for 10-fold cross-validation
train_control_gbm <- trainControl(method = "cv", 
                                  number = 10, 
                                  savePredictions = "final",
                                  allowParallel = TRUE, 
                                  verboseIter = TRUE)

# Specify hyperparameter tuning grid for GBM
tune_grid_gbm <- expand.grid(interaction.depth = c(1, 3, 5), 
                             n.trees = c(100, 200),
                             shrinkage = c(0.01, 0.1),
                             n.minobsinnode = c(10, 20))

# Train the model on the training set
gbm_model <- train(age ~ ., 
                   data = train_df_gbm, 
                   method = "gbm",
                   trControl = train_control_gbm,
                   tuneGrid = tune_grid_gbm,
                   metric = "MAE",
                   verbose = FALSE)

# Set up single round of cross-validation
repeats <- 1
folds <- 10

# Perform cross-validation on the test set
fold_list_gbm <- createFolds(test_df_gbm$age, k = folds, list = TRUE)

# Initialize vectors to store MAE, R² values, actual, and predicted values
mae_values_gbm <- numeric()
rsquared_values_gbm <- numeric()
actual_values_gbm <- numeric()
predicted_values_gbm <- numeric()

# Perform cross-validation and collect MAE and R² for each fold
for (fold in 1:folds) {
  test_indices_gbm <- fold_list_gbm[[fold]]
  test_fold_gbm <- test_df_gbm[test_indices_gbm, ]
  
  # Predict for the current fold
  predictions_gbm <- predict(gbm_model, newdata = test_fold_gbm[, -which(names(test_fold_gbm) == "age")])
  
  # Calculate MAE for the current fold
  mae_fold_gbm <- mean(abs(predictions_gbm - test_fold_gbm$age))
  mae_values_gbm <- c(mae_values_gbm, mae_fold_gbm)
  
  # Calculate R² for the current fold
  ss_total <- sum((test_fold_gbm$age - mean(test_fold_gbm$age))^2)
  ss_residual <- sum((test_fold_gbm$age - predictions_gbm)^2)
  rsquared_fold_gbm <- 1 - (ss_residual / ss_total)
  rsquared_values_gbm <- c(rsquared_values_gbm, rsquared_fold_gbm)
  
  # Store actual and predicted values for the jitter plot
  actual_values_gbm <- c(actual_values_gbm, test_fold_gbm$age)
  predicted_values_gbm <- c(predicted_values_gbm, predictions_gbm)
}

# Calculate the mean MAE and R² across all folds
mean_mae_gbm <- mean(mae_values_gbm)
mean_rsquared_gbm <- mean(rsquared_values_gbm)

# Print the mean MAE and R² on the test set
print(paste("Mean MAE on test set:", mean_mae_gbm))
print(paste("Mean R-squared on test set:", mean_rsquared_gbm))











################### Models trained on multi-omics dataset




# Set seed for reproducibility
set.seed(123)


# Split data into train and test sets
train_indices_combined <- createDataPartition(combined_filtered_df$age, p = 0.8, list = FALSE)
train_df_combined <- combined_filtered_df[train_indices_combined, ]
test_df_combined <- combined_filtered_df[-train_indices_combined, ]

# Specify train control for 10-fold cross-validation
train_control_combined <- trainControl(method = "cv", 
                                       number = 10, 
                                       savePredictions = "final",
                                       allowParallel = TRUE, 
                                       verboseIter = TRUE)

# Specify hyperparameter tuning grid
tune_grid_combined <- expand.grid(mtry = c(7, 135, 253, 364, 479, 547))

# Train the model on the training set
combined_rf_model <- train(age ~ ., 
                           data = train_df_combined, 
                           method = "rf",
                           trControl = train_control_combined,
                           tuneGrid = tune_grid_combined,
                           ntree = 400,
                           metric = "MAE")

# Set up single round of cross-validation
repeats <- 1
folds <- 10

# Perform cross-validation on the test set
fold_list_combined <- createFolds(test_df_combined$age, k = folds, list = TRUE)

# Initialize vectors to store MAE, R² values, actual, and predicted values
mae_values_combined <- numeric()
rsquared_values_combined <- numeric()
actual_values_combined <- numeric()
predicted_values_combined <- numeric()

# Perform cross-validation and collect MAE and R² for each fold
for (fold in 1:folds) {
  test_indices_combined <- fold_list_combined[[fold]]
  test_fold_combined <- test_df_combined[test_indices_combined, ]
  
  # Predict for the current fold
  predictions_combined <- predict(combined_rf_model, newdata = test_fold_combined[, -which(names(test_fold_combined) == "age")])
  
  # Calculate MAE for the current fold
  mae_fold_combined <- mean(abs(predictions_combined - test_fold_combined$age))
  mae_values_combined <- c(mae_values_combined, mae_fold_combined)
  
  # Calculate R² for the current fold
  ss_total <- sum((test_fold_combined$age - mean(test_fold_combined$age))^2)
  ss_residual <- sum((test_fold_combined$age - predictions_combined)^2)
  rsquared_fold_combined <- 1 - (ss_residual / ss_total)
  rsquared_values_combined <- c(rsquared_values_combined, rsquared_fold_combined)
  
  # Store actual and predicted values for the jitter plot
  actual_values_combined <- c(actual_values_combined, test_fold_combined$age)
  predicted_values_combined <- c(predicted_values_combined, predictions_combined)
}

# Calculate the mean MAE and R² across all folds
mean_mae_combined <- mean(mae_values_combined)
mean_rsquared_combined <- mean(rsquared_values_combined)

# Print the mean MAE and R² on the test set
print(paste("Mean MAE on test set:", mean_mae_combined))
print(paste("Mean R-squared on test set:", mean_rsquared_combined))









# Set seed for reproducibility
set.seed(123)


# Split data into train and test sets
train_indices_combined_xgb <- createDataPartition(combined_filtered_df$age, p = 0.8, list = FALSE)
train_combined_xgb <- combined_filtered_df[train_indices_combined_xgb, ]
test_combined_xgb <- combined_filtered_df[-train_indices_combined_xgb, ]

# Specify train control for 10-fold cross-validation
train_control_combined_xgb <- trainControl(method = "cv", 
                                           number = 10, 
                                           savePredictions = "final",
                                           allowParallel = TRUE, 
                                           verboseIter = TRUE)

# Specify hyperparameter tuning grid for XGBoost
tune_grid_combined_xgb <- expand.grid(nrounds = c(100, 200),
                                      max_depth = c(3, 6, 9),
                                      eta = c(0.01, 0.1, 0.3),
                                      gamma = c(0, 0.1, 0.3),
                                      colsample_bytree = 0.8,
                                      min_child_weight = 1,
                                      subsample = 0.8)

# Train the model on the training set
xgb_model_combined <- train(age ~ ., 
                            data = train_combined_xgb, 
                            method = "xgbTree",
                            trControl = train_control_combined_xgb,
                            tuneGrid = tune_grid_combined_xgb,
                            metric = "MAE")

# Set up single round of cross-validation
repeats <- 1
folds <- 10

# Perform cross-validation on the test set
fold_list_combined_xgb <- createFolds(test_combined_xgb$age, k = folds, list = TRUE)

# Initialize vectors to store MAE, R² values, actual, and predicted values
mae_values_combined_xgb <- numeric()
rsquared_values_combined_xgb <- numeric()
actual_values_combined_xgb <- numeric()
predicted_values_combined_xgb <- numeric()

# Perform cross-validation and collect MAE and R² for each fold
for (fold in 1:folds) {
  test_indices_combined_xgb <- fold_list_combined_xgb[[fold]]
  test_fold_combined_xgb <- test_combined_xgb[test_indices_combined_xgb, ]
  
  # Predict for the current fold
  predictions_combined_xgb <- predict(xgb_model_combined, newdata = test_fold_combined_xgb[, -which(names(test_fold_combined_xgb) == "age")])
  
  # Calculate MAE for the current fold
  mae_fold_combined_xgb <- mean(abs(predictions_combined_xgb - test_fold_combined_xgb$age))
  mae_values_combined_xgb <- c(mae_values_combined_xgb, mae_fold_combined_xgb)
  
  # Calculate R² for the current fold
  ss_total <- sum((test_fold_combined_xgb$age - mean(test_fold_combined_xgb$age))^2)
  ss_residual <- sum((test_fold_combined_xgb$age - predictions_combined_xgb)^2)
  rsquared_fold_combined_xgb <- 1 - (ss_residual / ss_total)
  rsquared_values_combined_xgb <- c(rsquared_values_combined_xgb, rsquared_fold_combined_xgb)
  
  # Store actual and predicted values for the jitter plot
  actual_values_combined_xgb <- c(actual_values_combined_xgb, test_fold_combined_xgb$age)
  predicted_values_combined_xgb <- c(predicted_values_combined_xgb, predictions_combined_xgb)
}

# Calculate the mean MAE and R² across all folds
mean_mae_combined_xgb <- mean(mae_values_combined_xgb)
mean_rsquared_combined_xgb <- mean(rsquared_values_combined_xgb)

# Print the mean MAE and R² on the test set
print(paste("Mean MAE on test set:", mean_mae_combined_xgb))
print(paste("Mean R-squared on test set:", mean_rsquared_combined_xgb))










# Set seed for reproducibility
set.seed(123)


# Split data into train and test sets
train_indices_combined_gbm <- createDataPartition(combined_filtered_df$age, p = 0.8, list = FALSE)
train_combined_gbm <- combined_filtered_df[train_indices_combined_gbm, ]
test_combined_gbm <- combined_filtered_df[-train_indices_combined_gbm, ]

# Specify train control for 10-fold cross-validation
train_control_combined_gbm <- trainControl(method = "cv", 
                                           number = 10, 
                                           savePredictions = "final",
                                           allowParallel = TRUE, 
                                           verboseIter = TRUE)

# Specify hyperparameter tuning grid for GBM
tune_grid_combined_gbm <- expand.grid(interaction.depth = c(2, 4, 6), 
                                      n.trees = c(150, 300),
                                      shrinkage = c(0.05, 0.2),
                                      n.minobsinnode = c(15, 30))

# Train the model on the training set
gbm_model_combined <- train(age ~ ., 
                            data = train_combined_gbm, 
                            method = "gbm",
                            trControl = train_control_combined_gbm,
                            tuneGrid = tune_grid_combined_gbm,
                            metric = "MAE",
                            verbose = FALSE)

# Set up single round of cross-validation
repeats_combined_gbm <- 1
folds_combined_gbm <- 10

# Perform cross-validation on the test set
fold_list_combined_gbm <- createFolds(test_combined_gbm$age, k = folds_combined_gbm, list = TRUE)

# Initialize vectors to store MAE, R² values, actual, and predicted values
mae_values_combined_gbm <- numeric()
rsquared_values_combined_gbm <- numeric()
actual_values_combined_gbm <- numeric()
predicted_values_combined_gbm <- numeric()

# Perform cross-validation and collect MAE and R² for each fold
for (fold in 1:folds_combined_gbm) {
  test_indices_combined_gbm <- fold_list_combined_gbm[[fold]]
  test_fold_combined_gbm <- test_combined_gbm[test_indices_combined_gbm, ]
  
  # Predict for the current fold
  predictions_combined_gbm <- predict(gbm_model_combined, newdata = test_fold_combined_gbm[, -which(names(test_fold_combined_gbm) == "age")])
  
  # Calculate MAE for the current fold
  mae_fold_combined_gbm <- mean(abs(predictions_combined_gbm - test_fold_combined_gbm$age))
  mae_values_combined_gbm <- c(mae_values_combined_gbm, mae_fold_combined_gbm)
  
  # Calculate R² for the current fold
  ss_total_combined_gbm <- sum((test_fold_combined_gbm$age - mean(test_fold_combined_gbm$age))^2)
  ss_residual_combined_gbm <- sum((test_fold_combined_gbm$age - predictions_combined_gbm)^2)
  rsquared_fold_combined_gbm <- 1 - (ss_residual_combined_gbm / ss_total_combined_gbm)
  rsquared_values_combined_gbm <- c(rsquared_values_combined_gbm, rsquared_fold_combined_gbm)
  
  # Store actual and predicted values for the jitter plot
  actual_values_combined_gbm <- c(actual_values_combined_gbm, test_fold_combined_gbm$age)
  predicted_values_combined_gbm <- c(predicted_values_combined_gbm, predictions_combined_gbm)
}

# Calculate the mean MAE and R² across all folds
mean_mae_combined_gbm <- mean(mae_values_combined_gbm)
mean_rsquared_combined_gbm <- mean(rsquared_values_combined_gbm)

# Print the mean MAE and R² on the test set
print(paste("Mean MAE on test set:", mean_mae_combined_gbm))
print(paste("Mean R-squared on test set:", mean_rsquared_combined_gbm))






########### Important feature ranking



# Feature importance for Random Forest models (using native randomForest package)
rf_importance <- randomForest::importance(rf_model$finalModel)
top_rf_importance <- head(rf_importance[order(-rf_importance[, 1]), ], 20)  # Sort and get top 20
print(top_rf_importance)

combined_rf_importance <- randomForest::importance(combined_rf_model$finalModel)
top_combined_rf_importance <- head(combined_rf_importance[order(-combined_rf_importance[, 1]), ], 20)
print(top_combined_rf_importance)

# Feature importance for XGBoost models (using native xgboost package)
xgb_importance <- xgb.importance(model = xgb_model$finalModel)
top_xgb_importance <- xgb_importance[1:20, ]  # Select top 20 features
print(top_xgb_importance)

combined_xgb_importance <- xgb.importance(model = xgb_model_combined$finalModel)
top_combined_xgb_importance <- combined_xgb_importance[1:20, ]  # Select top 20 features
print(top_combined_xgb_importance)

# Feature importance for GBM models (using native gbm package)
gbm_importance <- summary(gbm_model$finalModel)
top_gbm_importance <- gbm_importance[1:20, ]  # Select top 20 features
print(top_gbm_importance)

combined_gbm_importance <- summary(gbm_model_combined$finalModel)
top_combined_gbm_importance <- combined_gbm_importance[1:20, ]  # Select top 20 features
print(top_combined_gbm_importance)






################ IntegratedLearner model







# Exploring data dimensions
head(feature_table[1:5, 1:5])
head(sample_metadata[1:5, ])
head(feature_metadata[1:5, ])

# How many layers and how many features per layer?
table(feature_metadata$featureType)

# Number of subjects
length(unique(sample_metadata$subjectID))

# Sanity check
all(rownames(feature_table)==rownames(feature_metadata)) # TRUE
all(colnames(feature_table)==rownames(sample_metadata)) # TRUE




# Set seed for reproducibility
set.seed(123)

# Define the proportion of the data to be used for training
train_proportion <- 0.8

# Create the training and testing indices
train_indices <- createDataPartition(sample_metadata$Y, p = train_proportion, list = FALSE)

# Split the sample_metadata into training and testing sets
sample_metadata_train <- sample_metadata[train_indices, ]
sample_metadata_test <- sample_metadata[-train_indices, ]

# Extract the sample names for the training and testing sets
train_samples <- rownames(sample_metadata_train)
test_samples <- rownames(sample_metadata_test)

# Split the feature_table into training and testing sets based on the sample names
feature_table_train <- feature_table[, train_samples]
feature_table_test <- feature_table[, test_samples]

# feature_metadata remains the same since it contains feature-specific metadata
feature_metadata_train <- feature_metadata
feature_metadata_test <- feature_metadata

# Verifying the splits
cat("Dimensions of feature_table_train: ", dim(feature_table_train), "\n")
cat("Dimensions of feature_table_test: ", dim(feature_table_test), "\n")
cat("Dimensions of sample_metadata_train: ", dim(sample_metadata_train), "\n")
cat("Dimensions of sample_metadata_test: ", dim(sample_metadata_test), "\n")
cat("Dimensions of feature_metadata_train: ", dim(feature_metadata_train), "\n")
cat("Dimensions of feature_metadata_test: ", dim(feature_metadata_test), "\n")



#Run the model
fit<-IntegratedLearner(feature_table = feature_table_train,
                       sample_metadata = sample_metadata_train, 
                       feature_metadata = feature_metadata_train,
                       folds = 10,
                       base_learner = 'SL.BART',
                       meta_learner = 'SL.nnls',
                       verbose = TRUE)





plot.obj <- IntegratedLearner:::plot.learner(fit)

predict.learner <- function(fit,
                            feature_table_valid = NULL, # Feature table from validation set. Must have the exact same structure as feature_table.
                            sample_metadata_valid = NULL, # Optional: Sample-specific metadata table from independent validation set. Must have the exact same structure as sample_metadata.
                            feature_metadata = NULL) {
  
  # Check that feature names in fit object and feature_metadata match
  if (all(fit$feature.names == rownames(feature_metadata)) == FALSE) {
    stop("Both training feature_table and feature_metadata should have the same rownames.")
  }
  
  # Check if the feature_table_valid is provided
  if (is.null(feature_table_valid)) {
    stop("Feature table for validation set cannot be empty")
  }
  
  # Check that the feature names match between training and validation feature tables
  if (!is.null(feature_table_valid)) {
    if (all(fit$feature.names == rownames(feature_table_valid)) == FALSE) {
      stop("Both feature_table and feature_table_valid should have the same rownames.")
    }
  }
  
  # Check that the row names of sample_metadata_valid match the column names of feature_table_valid
  if (!is.null(sample_metadata_valid)) {
    if (all(colnames(feature_table_valid) == rownames(sample_metadata_valid)) == FALSE) {
      stop("Row names of sample_metadata_valid must match the column names of feature_table_valid")
    }
  }
  
  # Check for required columns in feature_metadata
  if (!'featureID' %in% colnames(feature_metadata)) {
    stop("feature_metadata must have a column named 'featureID' describing per-feature unique identifiers.")
  }
  
  if (!'featureType' %in% colnames(feature_metadata)) {
    stop("feature_metadata must have a column named 'featureType' describing the corresponding source layers.")
  }
  
  # Check for required columns in sample_metadata_valid
  if (!is.null(sample_metadata_valid)) {
    if (!'subjectID' %in% colnames(sample_metadata_valid)) {
      stop("sample_metadata_valid must have a column named 'subjectID' describing per-subject unique identifiers.")
    }
    
    if (!'Y' %in% colnames(sample_metadata_valid)) {
      stop("sample_metadata_valid must have a column named 'Y' describing the outcome of interest.")
    }
  }
  
  # Extract validation Y right away (will not be used anywhere during the validation process)
  validY <- NULL
  if (!is.null(sample_metadata_valid)) {
    validY <- sample_metadata_valid['Y']
  }
  
  # Stacked generalization input data preparation for validation data
  feature_metadata$featureType <- as.factor(feature_metadata$featureType)
  name_layers <- levels(droplevels(feature_metadata$featureType))
  
  X_test_layers <- vector("list", length(name_layers))
  names(X_test_layers) <- name_layers
  
  layer_wise_prediction_valid <- vector("list", length(name_layers))
  names(layer_wise_prediction_valid) <- name_layers
  
  for (i in seq_along(name_layers)) {
    # Prepare single-omic validation data and save predictions
    include_list <- feature_metadata %>% filter(featureType == name_layers[i])
    t_dat_slice_valid <- feature_table_valid[rownames(feature_table_valid) %in% include_list$featureID, ]
    dat_slice_valid <- as.data.frame(t(t_dat_slice_valid))
    X_test_layers[[i]] <- dat_slice_valid
    layer_wise_prediction_valid[[i]] <- predict.SuperLearner(fit$SL_fits$SL_fit_layers[[i]], newdata = dat_slice_valid)$pred
    rownames(layer_wise_prediction_valid[[i]]) <- rownames(dat_slice_valid)
  }
  
  combo_valid <- as.data.frame(do.call(cbind, layer_wise_prediction_valid))
  names(combo_valid) <- name_layers
  
  if (fit$run_stacked == TRUE) {
    stacked_prediction_valid <- predict.SuperLearner(fit$SL_fits$SL_fit_stacked, newdata = combo_valid)$pred
    rownames(stacked_prediction_valid) <- rownames(combo_valid)
  }
  if (fit$run_concat == TRUE) {
    fulldat_valid <- as.data.frame(t(feature_table_valid))
    concat_prediction_valid <- predict.SuperLearner(fit$SL_fits$SL_fit_concat, newdata = fulldat_valid)$pred
    rownames(concat_prediction_valid) <- rownames(fulldat_valid)
  }
  
  res <- list()
  
  if (!is.null(sample_metadata_valid)) {
    Y_test <- validY$Y
    res$Y_test <- Y_test
  }
  
  if (fit$run_concat & fit$run_stacked) {
    yhat.test <- cbind(combo_valid, stacked_prediction_valid, concat_prediction_valid)
    colnames(yhat.test) <- c(colnames(combo_valid), "stacked", "concatenated")
  } else if (fit$run_concat & !fit$run_stacked) {
    yhat.test <- cbind(combo_valid, concat_prediction_valid)
    colnames(yhat.test) <- c(colnames(combo_valid), "concatenated")
  } else if (!fit$run_concat & fit$run_stacked) {
    yhat.test <- cbind(combo_valid, stacked_prediction_valid)
    colnames(yhat.test) <- c(colnames(combo_valid), "stacked")
  } else {
    yhat.test <- combo_valid
  }
  
  res$yhat.test <- yhat.test
  
  if (!is.null(sample_metadata_valid)) {
    if (fit$family == 'binomial') {
      # Calculate AUC for each layer, stacked and concatenated
      pred <- apply(res$yhat.test, 2, ROCR::prediction, labels = res$Y_test)
      AUC <- vector(length = length(pred))
      names(AUC) <- names(pred)
      for (i in seq_along(pred)) {
        AUC[i] <- round(ROCR::performance(pred[[i]], "auc")@y.values[[1]], 3)
      }
      res$AUC.test <- AUC
    }
    
    if (fit$family == 'gaussian') {
      # Calculate R^2 for each layer, stacked and concatenated
      R2 <- vector(length = ncol(res$yhat.test))
      names(R2) <- names(res$yhat.test)
      for (i in seq_along(R2)) {
        R2[i] <- as.vector(cor(res$yhat.test[, i], res$Y_test)^2)
      }
      res$R2.test <- R2
    }
  }
  
  return(res)
}


# Set up cross-validation
repeats <- 1
folds <- 10
fold_list <- createFolds(sample_metadata_test$Y, k = folds, list = TRUE)

# Initialize lists to store results for each layer
results <- list()

# Perform cross-validation
for (fold in 1:folds) {
  # Subset data for the current fold
  test_indices <- fold_list[[fold]]
  feature_table_fold <- feature_table_test[, test_indices]
  sample_metadata_fold <- sample_metadata_test[test_indices, ]
  
  # Call predict.learner for the current fold
  predictions <- predict.learner(
    fit = fit,
    feature_table_valid = feature_table_fold,
    sample_metadata_valid = sample_metadata_fold,
    feature_metadata = feature_metadata
  )
  
  # Extract predicted and actual values for all layers
  predicted_all <- as.data.frame(predictions$yhat.test)
  actual_fold <- sample_metadata_fold$Y
  
  # Store results for all layers
  for (layer in colnames(predicted_all)) {
    if (!layer %in% names(results)) {
      results[[layer]] <- list(mae = numeric(), rsquared = numeric())
    }
    # Compute metrics for the current layer
    mae_layer <- mean(abs(predicted_all[[layer]] - actual_fold))
    rsquared_layer <- cor(predicted_all[[layer]], actual_fold)^2
    
    # Append metrics to the respective layer
    results[[layer]]$mae <- c(results[[layer]]$mae, mae_layer)
    results[[layer]]$rsquared <- c(results[[layer]]$rsquared, rsquared_layer)
  }
}














