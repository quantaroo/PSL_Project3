
# mymain.R
# -----------------------------------------------
# Sentiment Classification Model Training Script
# Processes a single train/test split using relative paths
# -----------------------------------------------

# Load required packages
suppressPackageStartupMessages({
  library(data.table)
  library(glmnet)
  library(pROC)
})

# Get command line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Please provide paths to train.csv and test.csv as arguments.")
}

# Assign paths from arguments
train_file <- args[1]
test_file  <- args[2]
output_file <- "mysubmission.csv"

# Load datasets
train_df <- fread(train_file)
test_df  <- fread(test_file)

# Extract features and labels
X_train <- as.matrix(train_df[, -(1:3)]) 
y_train <- train_df$sentiment
X_test <- as.matrix(test_df[, -(1:2)]) 

# Train logistic regression model with Lasso regularization
cv_fit <- cv.glmnet(X_train, y_train, family = "binomial", alpha = 1, nfolds = 5)
best_lambda <- cv_fit$lambda.min
model <- glmnet(X_train, y_train, family = "binomial", alpha = 1, lambda = best_lambda)

# Predict on the test set
pred_prob <- predict(model, X_test, type = "response")

# Save predictions to submission file
submission <- data.frame(id = test_df$id, prob = as.vector(pred_prob))
fwrite(submission, output_file)

# Save and Push Model to GitHub if processing split 1
if (grepl("split_1", train_file)) {
  model_file <- "trained_model.rds"
  saveRDS(model, file = model_file)
  
  # GitHub Push Logic
  repo_path <- "."  # Use current working directory
  repo <- repository(repo_path)
  
  # Stage the model file
  add(repo, model_file)
  
  # Commit changes
  commit(repo, message = "Updated trained_model.rds after split_1 training.")
  
  # Push changes
  push(repo, credentials = cred_user_pass(
    username = Sys.getenv("GITHUB_USERNAME"),
    password = Sys.getenv("GITHUB_PAT")
  ))
  
  cat("Model pushed to GitHub successfully!\n")
}
