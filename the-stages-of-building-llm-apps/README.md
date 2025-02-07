<h1>
  <span class="headline">Gen AI: LLM Capabilities and Alignment</span>
  <span class="subhead">The Stages of Building LLM Apps</span>
</h1>

**Learning Objective:** By the end of this lesson, you'll be able to explain the steps involved in building, fine-tuning, and deploying LLMs, including pretrained models, multimodal capabilities, and optimization challenges.

## Exploring Different Stages of Building LLM Applications
Large Language Models (LLMs) have become the backbone of numerous AI applications, but developing them requires an understanding of different methods and their associated stages. Below are the primary approaches to building LLM applications:

## Methods for Building LLM Applications
### Building LLMs from Scratch
- 🏛️ **Design a custom architecture** and train a model from the ground up.  
- 📊 **Collect massive datasets** and preprocess them for training.  
- 🖥️ **Train using distributed infrastructure** like GPUs/TPUs.  

### Using Pretrained Models
- 🔄 **Leverage existing models** like GPT, PaLM, or LLaMA via APIs or open-source platforms.  
- ⚡ **Reduce compute and training costs** by using already optimized models.  

### Fine-Tuning Pretrained Models
- 🔍 **Adapt a pretrained model** for specific use cases or domains using smaller datasets.  
- 🏆 **Enhance model performance** for specialized tasks like healthcare or legal analysis.  

### Building Multimodal LLMs
- 📸 **Develop models that process text, images, and audio** for diverse applications.  
- 🔗 **Enable cross-modal learning** by integrating various data sources.  


## Building LLMs from Scratch

Building an LLM from scratch is **resource-intensive but highly customizable**. It requires designing a custom architecture, collecting extensive datasets, and using large-scale infrastructure for training.

### Defining Architecture (e.g., Transformer)
- 🏛️ **Importance**: The architecture serves as the foundation for the model’s learning capabilities.  
- 🔹 **Key Decisions**:  
  - 🏗 **Selecting layers & attention heads**.  
  - ⚙️ **Configuring feedforward network sizes**.  
  - 🔡 **Defining tokenization techniques** (e.g., Byte-Pair Encoding).  
- 🧠 **Example**: The Transformer architecture, introduced in *Attention is All You Need*, powers models like **GPT and BERT**.  

### Collecting & Preprocessing Large Datasets 
- 🌍 **Data Sources**: Wikipedia, Common Crawl, domain-specific repositories.  
- 🧹 **Data Cleaning**: Remove duplicates, profanity, and irrelevant content.  
- 🏗 **Preprocessing**: Tokenize, normalize, and format for efficiency.  
- ⚖️ **Challenges**: Balancing dataset diversity, minimizing bias, and ensuring ethical compliance.  

---

### Training on Large-Scale Infrastructure
- 🚀 **Process**:  
  - 🖥️ Train on **distributed GPUs/TPUs** for efficiency.  
  - 🛠️ Use frameworks like **PyTorch, TensorFlow, or JAX**.  
  - 🎯 Optimize using **learning rate schedules & gradient clipping**.  
- 💾 **Infrastructure**:  
  - Requires **high-performance hardware** (e.g., NVIDIA A100 GPUs, TPU pods).  
  - ⏳ **Training takes weeks or months**, depending on dataset size.  
- 🌎 **Example**: GPT-3 was trained on **hundreds of petaflop/s-days of computation**.  

---

### Validating & Fine-Tuning for Task-Specific Accuracy
- 🔬 **Validation Metrics**: Perplexity, BLEU score, ROUGE, human evaluations.  
- ⚠️ **Detecting overfitting & underfitting**.  
- 🛠️ **Fine-Tuning Process**:  
  - Use **smaller, task-specific datasets** for performance improvement.  
  - Adjust **hyperparameters to optimize accuracy**.  

---

## Building Applications Using Pretrained LLMs

Pretrained LLMs provide **a practical and cost-effective way to build AI applications**, allowing **faster development, lower resources, and improved performance** across general and domain-specific tasks.

### Selecting Pretrained LLMs 
- 📏 **Considerations**:  
  - **Model Size**: **Larger models** (e.g., GPT-3) provide better performance but need more compute.  
  - **Task Suitability**: Ensure the model supports **text generation, translation, or summarization**.  
  - **Accessibility**:  
    - 🔗 **API-based models** (e.g., OpenAI GPT) provide easy access.  
    - 🖥️ **Open-source models** (e.g., LLaMA) allow **local deployment & customization**.  
- 🌟 **Popular Models**:  
  - **GPT** → Conversational AI, content creation.  
  - **PaLM** → Multilingual capabilities & knowledge-intensive applications.  
  - **LLaMA** → Lightweight, efficient for research & customization.  


###  Defining Application Goals

| 🏆 **Application**        | 🔍 **Goal**                                   | 🌍 **Example Use Case**                  |
|--------------------------|--------------------------------------------|------------------------------------------|
| 💬 **Chatbots**          | Generate **human-like responses**          | Virtual Assistants (e.g., Alexa)        |
| ✍️ **Content Generation** | Automate **blog writing, marketing copy** | AI-driven storytelling tools            |
| 📄 **Document Summarization** | Extract **key insights** from large texts | Legal contract summarization            |
| 📊 **Sentiment Analysis** | Detect **positive/negative tone** in text  | Brand reputation monitoring             |



### Integrating Pretrained LLMs via APIs or Libraries

| **Integration Method**          | **Advantages**                          | **Challenges**                         | **Use Case**                              |
|--------------------------------|--------------------------------------|-------------------------------------|--------------------------------------|
| **API-Based** (OpenAI, Cohere)  | ✅ **Easy setup & scalability** <br> ✅ **No infra required** | ⚠️ **Cost accumulates** <br> ⚠️ **Privacy concerns** | 🤖 **Customer support chatbots** |
| **Library-Based** (Hugging Face, TensorFlow) | ✅ **Customizable** <br> ✅ **Runs locally** | ⚠️ **Requires GPUs** <br> ⚠️ **More setup required** | 🏗️ **Fine-tuned legal document classifiers** |



## Fine-Tuning Pretrained LLMs 
Fine-tuning adapts **pretrained models** to **domain-specific needs**, optimizing **accuracy & efficiency**.  

### Key Steps in Fine-Tuning 

| **Step**                        | **Description**                                      | **Example**                               |
|--------------------------------|--------------------------------------------------|-------------------------------------------|
| **Choosing a Base Model**       | Select an LLM with suitable size & task compatibility. | 🏥 **BioBERT for medical texts**          |
| **Collecting Domain Data**      | Gather industry-specific datasets for fine-tuning.   | 💼 **Customer service chat logs**         |
| **Applying Fine-Tuning Techniques** | Train model on **smaller, task-specific datasets**.  | 📜 **Legal contract summarization**       |
| **Evaluating Performance**      | Validate using BLEU, ROUGE, F1 Score, or human feedback. | 🏆 **AI-generated financial reports** |


## Demonstration of LLM Application Build Stages

#### Step 1: Installing necessary libraries for working with LLMs, transformers, and vector embeddings
```python
!pip install torch transformers datasets sentence-transformers faiss-cpu
```

#### Step 2: Defining a minimal transformer-based architecture using PyTorch
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src):
        embedded = self.embedding(src)
        output = self.transformer(embedded, embedded)
        return self.fc(output)

# Example initialization
model = SimpleTransformer(vocab_size=5000, d_model=128, nhead=4, num_layers=2)
print("Custom Transformer model initialized.")
```

#### Step 3: Tokenizing & preprocessing a sample text dataset using the Hugging Face tokenizer
```python
from transformers import AutoTokenizer

# Load a tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Sample text dataset
text_samples = [
    "Artificial Intelligence is revolutionizing industries.",
    "LLMs are trained using massive datasets and powerful hardware.",
    "Fine-tuning enhances model performance for specific domains."
]

# Tokenizing the text
tokenized_texts = [tokenizer(text, padding="max_length", truncation=True, max_length=20) for text in text_samples]

print("Tokenized text sample:", tokenized_texts[0])
```

#### Step 4: Creating embeddings for tokenized text using Sentence Transformers
```python
from sentence_transformers import SentenceTransformer

# Load a sentence transformer model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings
text_embeddings = embedding_model.encode(text_samples)

print("Generated embeddings shape:", text_embeddings.shape)
```

#### Step 5: Setting Up a FAISS Vector Store for Text Retrieval
```python
import faiss
import numpy as np

# Create FAISS index
dimension = text_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# Add embeddings to the FAISS index
index.add(np.array(text_embeddings))

print(f"FAISS index initialized with {index.ntotal} stored vectors.")
```

#### Step 6: Performing Text Retrieval using FAISS similarity search
```python
query = "How does AI impact industries?"
query_embedding = embedding_model.encode([query])

# Retrieve the most relevant document
D, I = index.search(np.array(query_embedding), k=1)

print("Query:", query)
print("Most relevant document:", text_samples[I[0][0]])
```

#### Step 7: Using a Pretrained LLM (OpenAI's GPT-2 model) for Text Generation
```python
from transformers import pipeline

# Load GPT-2 model for text generation
generator = pipeline("text-generation", model="gpt2")

# Generate text based on a prompt
prompt = "The future of AI in business is"
response = generator(prompt, max_length=50, num_return_sequences=1)

print("Generated text:", response[0]["generated_text"])
```

#### Step 8: Preparing for Fine-Tuning a Pretrained LLM on a small custom dataset using the `Trainer` API
```python
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

# Load a sample dataset
dataset = load_dataset("imdb", split="train[:1000]")

# Load a pretrained model for fine-tuning
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Define training arguments
training_args = TrainingArguments(output_dir="./results", num_train_epochs=1, per_device_train_batch_size=8)

# Create Trainer object
trainer = Trainer(model=model, args=training_args, train_dataset=dataset)

print("Fine-tuning setup completed. Run trainer.train() to start training.")
```

#### Step 9: Evaluating Model Performance Using BLEU Score
```python
from nltk.translate.bleu_score import sentence_bleu

# Reference and candidate texts
reference = ["AI is transforming industries with automation."]
candidate = response[0]["generated_text"].split()

# Compute BLEU score
bleu_score = sentence_bleu([reference], candidate)
print("BLEU Score:", bleu_score)
```

## Building Multimodal LLMs
Multimodal LLMs **expand beyond text**, integrating **images, audio, and videos**.

### **1. Data Collection & Processing**  
- 📸 **Text-Image Datasets** → **MS COCO, OpenAI CLIP**.  
- 🎧 **Text-Audio Datasets** → **LibriSpeech**.  
- 🎥 **Video Analysis** → **YouTube-8M**.  

### **2. Architectures for Multimodal LLMs**  
- 🤖 **CLIP** → Learns text-image relationships.  
- 🎨 **DALL·E** → Generates images from text prompts.  
- 📡 **Multimodal Transformers** → Process multiple inputs simultaneously.  


## Key Takeaways 
✅ **Building from scratch is powerful but costly**.  
✅ **Pretrained LLMs allow rapid deployment with reduced compute needs**.  
✅ **Fine-tuning improves performance for specific applications**.  
✅ **Multimodal models enable richer AI capabilities**.  


## 🗣️ **Discussion Activity**  
1. What are the trade-offs between **using APIs vs. open-source LLMs**?  
2. How can fine-tuning be optimized for **low-resource domains**?  
3. What are the biggest challenges in **scaling multimodal LLMs**?  
