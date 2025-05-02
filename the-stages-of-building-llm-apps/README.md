<h1>
  <span class="headline">Gen AI: LLM Capabilities and Alignment</span>
  <span class="subhead">The Stages of Building LLM Apps</span>
</h1>

# The Stages of Building Large Language Models

## **Learning Objective**
By the end of this lesson, you will be able to:
-  **Explain** the steps involved in building, fine-tuning, and deploying LLMs
- **Assess** situations to decide when fine-tuning is necessary
- **Describe** how pretrained models and multimodal capabilities fit into different use cases, and how to address optimization challenges.

## **Understanding LLMs in the Machine Learning Landscape**

### **Where Do LLMs Fit?**
Large Language Models (LLMs) fall under the deep learning category of machine learning. They are trained on vast amounts of text data and can generate human-like responses. Here’s a **simplified hierarchy** of how LLMs fit within the ML ecosystem:

- **Machine Learning** → **Deep Learning** → **Transformer Models** → **LLMs (GPT, BERT, T5, etc.)**
- Unlike traditional NLP models, LLMs leverage **self-supervised learning** and transformer-based architectures for general-purpose text understanding and generation.

### **When Do You Need to Fine-Tune an LLM?**
Many organizations assume that fine-tuning is always required, but that’s not the case. Below is a **decision framework** for choosing between **fine-tuning, retrieval-augmented generation (RAG), or prompt engineering**:

<div class="mermaid">
graph TD
    A[Do you need to customize the model for a specific use case?] -->|No| B[Use prompt engineering]
    A -->|Yes| C{Do you need to incorporate private or proprietary knowledge?}
    
    C -->|No| D{Is your task highly specialized?}
    D -->|Yes| E[Fine-tune on domain-specific data]
    D -->|No| B

    C -->|Yes| F{Does retrieval work better than memorization?}
    F -->|Yes| G[Use RAG to fetch relevant information]
    F -->|No| E

    E --> H[Fine-Tune the LLM]
    B --> I[Optimize prompts for efficiency]
    G --> J[Build a RAG pipeline]
</div>

---

## **Hands-On Walkthrough: Fine-Tuning a Small LLM**

**Goal**: Fine-tune a **DistilBERT** model on the IMDB dataset and understand the basic workflow and optimizations, following the notebook's structure.

### **Step 1:** Install Dependencies
```python
# Installs necessary libraries from Hugging Face and PyTorch
!pip install transformers datasets torch --quiet
```

### **Step 2:** Load a Pretrained Model and Dataset
```python
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from PIL import Image
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    CLIPModel,         # Added for multimodal section later
    CLIPProcessor      # Added for multimodal section later
)
from datasets import load_dataset
import requests        # Added for multimodal section later
from io import BytesIO # Added for multimodal section later


# Load dataset and model
dataset = load_dataset("imdb")  # Standard benchmark dataset for sentiment analysis
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2) # num_labels=2 for positive/negative sentiment
```

### **Step 3:** (Optional Optimization) Freeze Initial Layers
*This step appears after loading the model but before data preprocessing in the notebook. Freezing initial layers (like embeddings) can speed up training by reducing the number of parameters to update, especially useful if compute is limited or you want the model to retain its general language understanding.*
```python
# Speed optimization 1: Freeze embedding layers
for param in model.distilbert.embeddings.parameters():
    param.requires_grad = False
```

### **Step 4:** Preprocess the Data
*Tokenization converts raw text into numerical IDs the model understands. We also select a smaller subset for faster demonstration.*
```python
def tokenize_function(examples):
    # Tokenize text, pad shorter sequences to max_length, and truncate longer ones.
    # max_length=128 is chosen for speed; longer sequences might capture more context but take more memory/time.
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128) # Speed optimization 2: Limit max length

# Apply the tokenization function to the entire dataset in batches for efficiency.
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Select smaller subsets for faster training and evaluation during the demo.
# Shuffling ensures the subset is representative.
train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(500))  # Speed optimization 3: Even smaller dataset
eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(50))  # Reduced evaluation set (used later)
```

### **Step 5:** Configure and Run Fine-Tuning using `Trainer` API
*The `Trainer` API simplifies the training loop. We configure it with optimized arguments and then start the training.*
```python
# Configuration for optimized training using the Trainer API
training_args = TrainingArguments(
    output_dir="./results",  # Directory for checkpoints and logs

    # Training strategy settings
    eval_strategy="no",  # Disable evaluation during training for speed (use "epoch" or "steps" for evaluation)
    num_train_epochs=1,  # Reduced epochs for faster demo

    # Batch size and gradient accumulation
    per_device_train_batch_size=16, # Batch size per GPU/CPU
    gradient_accumulation_steps=2,  # Update weights every 2 batches (effective batch size = 16*2=32)

    # Optimization settings
    fp16=True if torch.cuda.is_available() else False,  # Use mixed precision (faster, less memory) if CUDA is available

    # Logging and reporting
    report_to="none",  # Disable external logging (like Weights & Biases)
    logging_steps=10,  # Log training loss every 10 steps
)

# Create trainer instance using the prepared model and tokenized dataset
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
    # eval_dataset is omitted because eval_strategy="no"
)

# Start the fine-tuning process
trainer.train()
```

### **Step 6:** Test the Fine-Tuned Model
```python
# Test the model with a sample input
inputs = tokenizer("This movie was absolutely fantastic!", return_tensors="pt")

# Get model predictions (logits)
# Note: The Trainer automatically handles device placement during training.
# For inference after Trainer, ensure inputs are on the same device as the model.
model.to("cpu") # Move model back to CPU if needed, or keep on GPU if available
inputs = {k: v.to(model.device) for k, v in inputs.items()} # Move inputs to model's device

outputs = model(**inputs)
print("Logits for 'fantastic':", outputs.logits) # Higher value for the positive class (index 1 probably) expected

# Test with a negative example
inputs_neg = tokenizer("This movie was terrible.", return_tensors="pt")
inputs_neg = {k: v.to(model.device) for k, v in inputs_neg.items()} # Move to model's device
outputs_neg = model(**inputs_neg)
print("Logits for 'terrible':", outputs_neg.logits) # Higher value for the negative class (index 0 probably) expected
```
- **Reflection**: Did the fine-tuning process adjust the model's predictions? Compare the logits for the positive and negative examples. (Note: A single epoch on a small subset might show only slight changes).

---

## **Hands-On Optimization Task: Debugging Training Inefficiencies**

Now, let’s look at a manual training loop and identify inefficiencies compared to the `Trainer` API or an optimized manual loop. *This section demonstrates common pitfalls.*

```python
# Re-define an optimizer for this specific manual loop demonstration
# (Trainer manages its own optimizer internally)
# Ensure model parameters requiring gradients are included (embeddings might still be frozen from Step 3)
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = AdamW(trainable_params, lr=5e-5)

# INEFFICIENT training process example (as shown in notebook)
# Problems highlighted:
# 1. Iterates sample by sample (no batching via DataLoader).
# 2. Tokenizes *inside* the loop (redundant computation).
# 3. No explicit device management (runs on CPU unless manually moved).
# 4. Fixed number of epochs might be too much without validation.
# 5. Uses the `eval_dataset` for iteration demonstration - normally you'd iterate `train_dataset`.
print("\n--- Starting Inefficient Loop Demo (will be slow) ---")
num_epochs_inefficient = 1 # Reduced drastically for demo speed
inefficient_counter = 0
max_inefficient_steps = 10 # Limit steps for demo

model.to("cpu") # Ensure model is on CPU for this specific inefficient example if GPU was used before

for epoch in range(num_epochs_inefficient):
    print(f"Inefficient Epoch {epoch+1}/{num_epochs_inefficient}")
    # Using eval_dataset just to have a small iterable; normally use train_dataset
    for batch_item in eval_dataset: # INEFFICIENT: Iterating single items
        if inefficient_counter >= max_inefficient_steps:
             break

        # INEFFICIENT: Repeated tokenization inside loop for each example
        # Ensure padding/truncation match the efficient approach for fair comparison if running longer
        inputs = tokenizer(batch_item["text"], return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs['labels'] = torch.tensor([batch_item["label"]]) # Create label tensor

        # INEFFICIENT: No explicit device placement (will run on CPU here)
        outputs = model(**inputs)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad() # Correct placement for single item update

        inefficient_counter += 1
        if inefficient_counter % 5 == 0: # Log progress infrequently
            print(f"  Inefficient step {inefficient_counter}, Loss: {loss.item()}")

    if inefficient_counter >= max_inefficient_steps:
        print(f"Reached max {max_inefficient_steps} inefficient steps for demo.")
        break
print("--- Finished Inefficient Loop Demo ---")

```

<details>
  <summary><strong>🤔 What’s wrong with the loop above?</strong></summary>

  - **No Batching:** Processing examples one by one is extremely slow compared to batch processing. GPUs excel at parallel computation on batches.
  - **Repeated Tokenization:** Tokenizing should be done once as a preprocessing step, not repeatedly inside the training loop.
  - **No Device Management:** The code doesn't explicitly move the model or data to the GPU (if available), missing potential speedups.
  - **Inefficient Iteration:** Directly iterating over a `Dataset` object is less efficient than using a `DataLoader`, which handles batching, shuffling, and parallel data loading.
  - **Epoch Count:** Running for many epochs on small data without validation can lead to overfitting quickly.

</details>

### **Optimized Manual Loop Version**
*This version uses `DataLoader` for batching and handles device placement, mirroring best practices without the `Trainer` abstraction.*

```python
from torch.utils.data import DataLoader

# Create DataLoader for efficient batching and shuffling
# Use the actual small train_dataset created in Step 4
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Fresh optimizer for this specific loop
# Ensure we capture parameters that are currently trainable
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = AdamW(trainable_params, lr=5e-5) # Reinitialize optimizer state

# Determine device (CPU or GPU) and move model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("\n--- Starting Efficient Loop Demo ---")
num_epochs_efficient = 1 # Reduced epochs appropriate for small dataset/demo

for epoch in range(num_epochs_efficient):
    print(f"Efficient Epoch {epoch+1}/{num_epochs_efficient}")
    model.train() # Set model to training mode
    step_counter = 0
    for batch in train_dataloader: # EFFICIENT: Iterating over batches from DataLoader
        optimizer.zero_grad() # Zero gradients *before* the batch processing

        # Prepare inputs (assuming batch contains 'text' and 'label' from original dataset structure)
        # This assumes the DataLoader is yielding dictionaries with these keys
        # In a real scenario with pre-tokenized data, keys would be 'input_ids', 'attention_mask', etc.
        inputs = tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True, max_length=128) # Tokenize the batch text
        labels = batch["label"] # Get labels from the batch
        inputs['labels'] = labels # Add labels to the inputs dictionary

        # EFFICIENT: Move batch data to the same device as the model
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass
        outputs = model(**inputs)
        loss = outputs.loss

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

        step_counter += 1
        if step_counter % 10 == 0: # Log every 10 steps
             print(f"  Efficient step {step_counter}, Loss: {loss.item()}") # Log loss

print("--- Finished Efficient Loop Demo ---")
```

## **Exploring Multimodal LLMs**

### **Why Fine-Tuning Isn't Always Enough**
- Some tasks require understanding relationships between different types of data (e.g., describing an image, finding images matching text).
- **Fine-tuning a text-only LLM won’t inherently teach it to understand images or audio.**
- **Multimodal models** like CLIP are pretrained on paired data (e.g., images and their captions) to learn joint embeddings.

### **Quick Code Demo: Image + Text Model (CLIP)**
*CLIP learns to map images and text descriptions into a shared embedding space, allowing similarity comparisons.*
```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests # To fetch image from URL
from io import BytesIO # To handle image bytes

# Load pretrained CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# URL of an image (replace with a working URL or use the example)
# Example: A picture of cats playing
image_url = "https://images.pexels.com/photos/161735/pexels-photo-161735.jpeg"
# Example: A picture of a dog
# image_url = "https://images.pexels.com/photos/1108099/pexels-photo-1108099.jpeg"

print(f"\n--- Starting CLIP Demo ---")
try:
    # Fetch the image from the URL
    print(f"Fetching image from: {image_url}")
    response = requests.get(image_url)
    response.raise_for_status() # Check if the request was successful
    image = Image.open(BytesIO(response.content))
    print("Image fetched successfully.")

    # Process the image and text prompts
    # Using slightly different prompts than the notebook for clarity
    text_prompts = ["a photo of cats playing", "a photo of a dog running"]
    print(f"Processing image with text prompts: {text_prompts}")
    inputs = processor(text=text_prompts, images=image, return_tensors="pt", padding=True)

    # Get similarity scores from the model
    # Move model to appropriate device if necessary (CLIP models are large)
    clip_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model.to(clip_device)
    inputs = {k: v.to(clip_device) for k, v in inputs.items()}

    with torch.no_grad(): # Disable gradient calculation for inference
        outputs = clip_model(**inputs)

    logits_per_image = outputs.logits_per_image # Logits representing similarity
    probs = logits_per_image.softmax(dim=1) # Convert logits to probabilities

    print(f"\nResults for image: {image_url}")
    print(f"Similarity Logits (Image vs {text_prompts}): {logits_per_image.cpu().numpy()}")
    print(f"Similarity Probabilities (Image vs {text_prompts}): {probs.cpu().numpy()}")

except Exception as e:
    print(f"Error processing image from URL: {e}")
    print("Please ensure the image_url is valid and accessible.")
print("--- Finished CLIP Demo ---")
```
*Expected output: Higher probability for the text prompt that correctly describes the image.*

## **Bridging to RAG (Retrieval-Augmented Generation)**

Now that we’ve explored **fine-tuning, optimization, and multimodal models**, let’s consider scenarios where these might still fall short, particularly regarding factual knowledge:

> **If fine-tuning struggles with keeping knowledge up-to-date or handling vast amounts of specific facts, how can we improve LLM responses?**

- Fine-tuning primarily adapts the model's *style* or *task-specific behavior*. It's **not efficient or reliable for memorizing large, constantly changing factual databases**.
- Instead of forcing an LLM to memorize everything (which fine-tuning attempts implicitly), **Retrieval-Augmented Generation (RAG)** provides the model with relevant, up-to-date information *at inference time*. The LLM then uses this retrieved context to generate a more accurate and informed response. This is often more effective for knowledge-intensive tasks.

---
