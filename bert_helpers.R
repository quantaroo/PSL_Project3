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

# Import Python Packages
transformers <- reticulate::import("transformers")
torch <- reticulate::import("torch")

# Use CUDA if available, otherwise CPU
device <- if (torch$cuda$is_available()) torch$device("cuda") else torch$device("cpu")
cat(sprintf("Using device: %s\n", device$`type`))

# Load BERT Model and Tokenizer
cat("Loading BERT model...\n")
tokenizer <- transformers$AutoTokenizer$from_pretrained("distilbert-base-uncased")
model_bert <- transformers$AutoModel$from_pretrained("distilbert-base-uncased")$to(device)

# Function for Processing Text in Batches
get_bert_embeddings_batch <- function(texts, batch_size = 100) {
  total_texts <- length(texts)
  all_embeddings <- list()
  start_time <- Sys.time()
  
  for (i in seq(1, total_texts, by = batch_size)) {
    # Comment out the progress messages
    # cat(sprintf("Processing batch %d - %d of %d\n", i, min(i + batch_size - 1, total_texts), total_texts))
    
    batch_texts <- texts[i:min(i + batch_size - 1, total_texts)]
    texts_py <- r_to_py(as.list(as.character(batch_texts)))
    
    inputs <- tokenizer$batch_encode_plus(
      texts_py, 
      return_tensors = "pt", 
      padding = TRUE, 
      truncation = TRUE
    )
    
    with(torch$no_grad(), {
      outputs <- model_bert$forward(
        input_ids = inputs$input_ids$to(device), 
        attention_mask = inputs$attention_mask$to(device)
      )
    })
    
    embeddings <- outputs$last_hidden_state$mean(dim = 2L)$detach()
    all_embeddings[[length(all_embeddings) + 1]] <- as.matrix(embeddings$cpu()$numpy())

    # Memory cleanup
    if (torch$cuda$is_available()) torch$cuda$empty_cache()
  }
  
  # Combine all batches
  do.call(rbind, all_embeddings)
}

# Function to Align Embeddings to Target Dimension
align_embeddings <- function(embeddings, target_dim = 1536) {
  current_dim <- ncol(embeddings)
  if (current_dim < target_dim) {
    # Pad embeddings with zeros
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
  embeddings <- get_bert_embeddings_batch(newdata, batch_size = 100)
  aligned_embeddings <- align_embeddings(embeddings, target_dim = 1536)
  predict(model, aligned_embeddings, type = "response")
}
