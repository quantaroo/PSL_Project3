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

# Function to Load Python Packages
load_python_packages <- function() {
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
  
  list(transformers = transformers, torch = torch)
}

# Load Packages Once
python_packages <- load_python_packages()

# Set Device to CUDA if Available
device <- if (python_packages$torch$cuda$is_available()) {
  python_packages$torch$device("cuda")
} else {
  python_packages$torch$device("cpu")
}
cat(sprintf("Using device: %s\n", device$`type`))

# Function to Load BERT Model
load_bert_model <- function(model_name = "distilbert-base-uncased") {
  cat("Loading BERT model...\n")
  
  tokenizer <- python_packages$transformers$AutoTokenizer$from_pretrained(model_name)
  model_bert <- python_packages$transformers$AutoModel$from_pretrained(model_name)$to(device)
  
  list(tokenizer = tokenizer, model_bert = model_bert)
}

# Load Model Once
bert_model <- load_bert_model()

# Function for Processing Text in Batches
get_bert_embeddings_batch <- function(texts, batch_size = 100) {
  total_texts <- length(texts)
  all_embeddings <- list()
  
  for (i in seq(1, total_texts, by = batch_size)) {
    # Select Batch
    batch_texts <- texts[i:min(i + batch_size - 1, total_texts)]
    texts_py <- r_to_py(as.list(as.character(batch_texts)))
    
    # Tokenize Text
    inputs <- bert_model$tokenizer$batch_encode_plus(
      texts_py, 
      return_tensors = "pt", 
      padding = TRUE, 
      truncation = TRUE
    )
    
    # Forward Pass Through BERT
    with(python_packages$torch$no_grad(), {
      outputs <- bert_model$model_bert$forward(
        input_ids = inputs$input_ids$to(device), 
        attention_mask = inputs$attention_mask$to(device)
      )
    })
    
    # Extract Embeddings and Detach Tensors
    embeddings <- outputs$last_hidden_state$mean(dim = 2L)$detach()
    all_embeddings[[length(all_embeddings) + 1]] <- as.matrix(embeddings$cpu()$numpy())

    # Clear GPU Memory if CUDA is Available
    if (python_packages$torch$cuda$is_available()) python_packages$torch$cuda$empty_cache()
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
  embeddings <- get_bert_embeddings_batch(newdata, batch_size = 100)
  aligned_embeddings <- align_embeddings(embeddings, target_dim = 1536)
  predict(model, aligned_embeddings, type = "response")
}
