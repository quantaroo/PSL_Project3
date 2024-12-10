# bert_helpers.R
# -------------------------------------------
# Helper Functions for BERT Embeddings and Prediction
# -------------------------------------------

# Load required packages
suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(reticulate)
})

# Set up the Python environment dynamically
tryCatch({
  reticulate::use_condaenv("bert_env", required = TRUE)
}, error = function(e) {
  cat("Conda environment 'bert_env' not found. Defaulting to system Python.\n")
})

# Configure Reticulate
reticulate::py_config()

# Import Python Packages Safely
transformers <- tryCatch(
  reticulate::import("transformers"),
  error = function(e) stop("Error: Could not load Hugging Face Transformers.")
)

torch <- tryCatch(
  reticulate::import("torch"),
  error = function(e) stop("Error: Could not load PyTorch.")
)

# Set Device to CUDA if Available
device <- if (torch$cuda$is_available()) torch$device("cuda") else torch$device("cpu")
cat(sprintf("Using device: %s\n", device$`type`))

# Load BERT Model and Tokenizer
cat("Loading BERT model...\n")
tokenizer <- transformers$AutoTokenizer$from_pretrained("distilbert-base-uncased")
model_bert <- transformers$AutoModel$from_pretrained("distilbert-base-uncased")$to(device)

# Updated Function: Padding Embeddings Safely
get_bert_embeddings_batch <- function(texts, batch_size = 50, target_dim = 1536) {
  total_texts <- length(texts)
  all_embeddings <- list()
  start_time <- Sys.time()
  
  for (i in seq(1, total_texts, by = batch_size)) {
    cat(sprintf("Processing batch %d - %d of %d\n", i, min(i + batch_size - 1, total_texts), total_texts))
    
    # Extract Batch Texts
    batch_texts <- texts[i:min(i + batch_size - 1, total_texts)]
    texts_py <- r_to_py(as.list(as.character(batch_texts)))
    
    # Tokenize the Texts
    inputs <- tokenizer$batch_encode_plus(
      texts_py, 
      return_tensors = "pt", 
      padding = TRUE, 
      truncation = TRUE
    )
    
    # Forward Pass Through BERT Model
    with(torch$no_grad(), {
      outputs <- model_bert$forward(
        input_ids = inputs$input_ids$to(device), 
        attention_mask = inputs$attention_mask$to(device)
      )
    })
    
    # Extract Embeddings
    embeddings <- outputs$last_hidden_state$mean(dim = 2L)$detach()
    embeddings_matrix <- as.matrix(embeddings$cpu()$numpy())
    
    # Ensure Correct Dimensions by Padding or Truncating
    if (ncol(embeddings_matrix) != target_dim) {
      embeddings_matrix <- align_embeddings(embeddings_matrix, target_dim)
    }
    
    # Append Embeddings
    all_embeddings[[length(all_embeddings) + 1]] <- embeddings_matrix
    
    # Clear CUDA Memory
    if (torch$cuda$is_available()) torch$cuda$empty_cache()
    
    # Print Progress Estimate
    elapsed_time <- Sys.time() - start_time
    batches_done <- ceiling(i / batch_size)
    batches_left <- ceiling(total_texts / batch_size) - batches_done
    avg_batch_time <- elapsed_time / batches_done
    estimated_time_left <- avg_batch_time * batches_left
    
    cat(sprintf("Estimated time left: ~%.2f minutes\n", as.numeric(estimated_time_left, units = "mins")))
  }

  # Combine All Batches
  do.call(rbind, all_embeddings)
}

# Function to Align Embeddings to Target Dimension
align_embeddings <- function(embeddings, target_dim = 1536) {
  current_dim <- ncol(embeddings)
  if (current_dim < target_dim) {
    # Pad Embeddings with Zeros
    cbind(embeddings, matrix(0, nrow = nrow(embeddings), ncol = target_dim - current_dim))
  } else {
    embeddings[, 1:target_dim]
  }
}

# Function to Validate Input Data
check_input_data <- function(newdata) {
  if (!is.character(newdata) || length(newdata) == 0) {
    stop("Input data must be a non-empty character vector.")
  }
}

# Prediction Function
predict_function <- function(model, newdata) {
  check_input_data(newdata)
  embeddings <- get_bert_embeddings_batch(newdata, batch_size = 50)
  aligned_embeddings <- align_embeddings(embeddings, target_dim = 1536)
  predict(model, aligned_embeddings, type = "response")
}
