<h1>
  <span class="headline">Gen AI: LLM Capabilities and Alignment</span>
  <span class="subhead">Introduction to RAG</span>
</h1>

**Learning Objective:** By the end of this lesson, you'll be able to explain Retrieval-Augmented Generation (RAG), its workflow, components, applications, and evaluation for enhancing AI knowledge retrieval.

## An Introduction to RAG
Retrieval-Augmented Generation (RAG) is a framework that combines information retrieval systems with generative language models. By dynamically fetching relevant external knowledge, RAG enhances language models' performance, making them more **accurate, contextual, and adaptable** for knowledge-intensive tasks.

### What is RAG? 
- RAG integrates a **retrieval module** to fetch external knowledge and a **generative module** to synthesize responses based on the retrieved data.  
- Unlike traditional LLMs, which rely solely on static training data, RAG systems dynamically update their knowledge by accessing real-time or external information.  

### How RAG Works 
1. **Input Query**: A user query is processed by the retrieval module.  
2. **Document Retrieval**: Relevant documents or embeddings are fetched from external databases (e.g., vector databases like Pinecone, FAISS).  
3. **Response Generation**: The generative model uses the retrieved data to create **accurate, context-aware responses**.  

### Why RAG Matters
- 📚 **Reduces dependency** on model parameters for storing knowledge, making the system **more scalable and adaptable**.  
- 🤖 **Enhances performance** for **knowledge-intensive applications**, such as customer support, summarization, and research assistance.

## Components of RAG
### 🔎 **1. Retrieval Module**  
- 📂 Fetches relevant documents or embeddings from external sources using **semantic similarity**.  
- 📊 Uses **vector databases** (e.g., Pinecone, Weaviate) to store and query document embeddings.  
- 🔍 Compares **query embeddings** with stored embeddings to identify the **most relevant matches**.  

### ✍️ **2. Generative Module**  
- 🧠 **Synthesizes responses** by combining retrieved information with pre-trained language capabilities.  
- ⚙️ **Examples of generative models**: **GPT, T5, PaLM**.  


## How RAG Differs from Traditional LLMs

| Aspect             | Traditional LLMs                             | RAG                                          |
|--------------------|--------------------------------------------|----------------------------------------------|
| 📚 **Knowledge Source**  | Static training data                      | Dynamically retrieves external knowledge    |
| 📈 **Scalability**       | Limited by model size                    | Scalable due to external knowledge access  |
| 🔄 **Updates**           | Requires retraining to update knowledge  | Retrieves real-time information dynamically |
| ⚠️ **Bias Mitigation**   | Susceptible to biases in training data   | Allows retrieval of more diverse perspectives |



## Step-by-Step Workflow of RAG
1. **Input Query**: A user submits a query (e.g., "What are the symptoms of diabetes?").  
2. **Document Retrieval**: The **retrieval module** searches external knowledge bases or vector databases for **relevant documents**.  
3. **Response Generation**: The **generative module** synthesizes a response using both the query and retrieved knowledge.  
4. **Final Output**: A **coherent, fact-based response** is provided to the user.  

### **Challenges in Implementation**
- 🔍 **Retrieval Accuracy**: Ensuring only **high-quality and relevant** documents are retrieved.  
- 📉 **Noise in Retrieved Data**: Poor-quality or irrelevant data may degrade response quality.  
- ⏳ **Latency**: Combining retrieval and generation can slow down response times.  

## Demonstration of a RAG Implementation
#### Step 1: Installing Required Libraries
```python
!pip install transformers faiss-cpu sentence-transformers datasets
```

#### Step 2: Loading Pretrained Sentence Transformer for Embedding Generation
```python
from sentence_transformers import SentenceTransformer

# Load a pretrained embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded successfully.")
```

#### Step 3: Generating Embeddings for Sample Documents
```python
documents = [
    "Artificial Intelligence is transforming industries worldwide.",
    "Machine learning enables computers to learn from data patterns.",
    "Retrieval-Augmented Generation (RAG) enhances generative AI with external knowledge.",
    "Vector databases efficiently store and retrieve high-dimensional embeddings.",
    "OpenAI’s GPT models are widely used for text generation."
]

# Generate embeddings
document_embeddings = embedding_model.encode(documents)

print("Generated embeddings shape:", document_embeddings.shape)
```

#### Step 4: Setting Up a FAISS Vector Database for Retrieval
```python
import faiss
import numpy as np

# Create FAISS index
dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# Add embeddings to the index
index.add(np.array(document_embeddings))

print(f"FAISS index initialized with {index.ntotal} vectors.")
```

#### Step 5: Retrieving Relevant Documents Using FAISS
```python
query = "How does AI use external knowledge?"
query_embedding = embedding_model.encode([query])

# Retrieve top-1 closest document
D, I = index.search(np.array(query_embedding), k=1)

print("Query:", query)
print("Most relevant document:", documents[I[0][0]])
```

#### Step 6: Loading a Pretrained Generative Model (GPT-2)
```python
from transformers import pipeline

# Load a text generation model
generator = pipeline("text-generation", model="gpt2")

print("Generative model loaded successfully.")
```

#### Step 7: Generating a Response Using Retrieved Knowledge
```python
context = documents[I[0][0]]
prompt = f"Based on the information: '{context}', explain how AI retrieves external knowledge."

# Generate response
response = generator(prompt, max_length=100, num_return_sequences=1)

print("Generated response:", response[0]["generated_text"])
```

#### Step 8: Evaluating the Generated Response with BLEU Score
```python
from nltk.translate.bleu_score import sentence_bleu

# Reference and generated text
reference = ["AI retrieves external knowledge by accessing external databases."]
candidate = response[0]["generated_text"].split()

# Compute BLEU score
bleu_score = sentence_bleu([reference], candidate)
print("BLEU Score:", bleu_score)
```

#### Step 9: Performing a Similarity Check Between Query and Retrieved Data
```python
from sklearn.metrics.pairwise import cosine_similarity

# Compute cosine similarity between query and retrieved document
similarity_score = cosine_similarity(query_embedding, [document_embeddings[I[0][0]]])

print("Cosine Similarity Score:", similarity_score[0][0])
```

## Applications of RAG
| Application                      | How RAG Helps                                   | Example Use Case                                      |
|----------------------------------|-----------------------------------------------|------------------------------------------------------|
| 💬 **Chatbots & Virtual Assistants**  | Retrieves real-time information for customer support. | Virtual assistants answering FAQs using company databases. |
| 📄 **Document Summarization**       | Extracts key points from lengthy documents.  | Legal firms summarizing case law.                     |
| ⚖️ **Domain-Specific Retrieval**    | Accesses industry-specific knowledge.         | Healthcare AI retrieving the latest medical research. |
| 🏢 **Enterprise Knowledge Management** | Retrieves and organizes company data.         | HR chatbots fetching policy documents.                |
| 🌍 **Multilingual Support**          | Generates responses in multiple languages.    | E-commerce support in various languages.             |


## Role of Vector Databases in RAG
### **1. Embedding Storage and Similarity-Based Retrieval**
- 🔢 **What are Embeddings?** Numerical representations of data that encode semantic meaning.  
- 🔎 **How Retrieval Works**:  
  - 🔄 **Query Conversion**: The user query is converted into an embedding.  
  - 📊 **Similarity Search**: The query embedding is compared with stored embeddings using **cosine similarity**.  
  - 🎯 **Top Matches**: The most relevant embeddings are retrieved and passed to the **generative module**.  


## Evaluating RAG Systems
### **1. Metrics for Retrieval Effectiveness**
| Metric               | Description                                       | Use Case                                      |
|----------------------|--------------------------------------------------|----------------------------------------------|
| 🎯 **Precision**       | Measures how many retrieved documents are relevant. | Legal research retrieval.                    |
| 🔄 **Recall**         | Measures how many relevant documents were retrieved. | Healthcare AI ensuring critical research isn’t missed. |
| 📊 **Embedding Similarity** | Measures semantic similarity between query and retrieved data. | Customer support retrieving FAQ answers.     |

### **2. Assessing Generative Quality**
| Metric          | Description                                      | Use Case                     |
|----------------|------------------------------------------------|------------------------------|
| 🔵 **BLEU**    | Measures similarity of generated text to reference text. | Machine translation.        |
| 🔴 **ROUGE**   | Evaluates recall of key phrases in summaries.   | Document summarization.     |
| 📉 **Perplexity** | Measures fluency and coherence of generated text. | Chatbot interactions.       |




## Real-World Testing & Robustness
- 💥 **Stress Testing**: Evaluating system performance under **high demand**.  
- 🛡️ **Robustness Checks**: Handling **ambiguous, incomplete, or noisy** queries.  
- 🎭 **Edge Case Testing**: Ensuring AI can manage **unexpected inputs**.  


## Conclusion & Key Takeaways
1️⃣ **Retrieval-Augmented Generation (RAG) enhances knowledge retrieval** by dynamically accessing external sources.  
2️⃣ **Vector databases play a critical role** in managing and retrieving embeddings efficiently.  
3️⃣ **Evaluating RAG performance** requires multiple metrics, including **precision, recall, BLEU, and ROUGE**.  
4️⃣ **Scalability, latency, and retrieval accuracy** are key challenges when deploying RAG.  
5️⃣ **Real-world testing ensures robustness** and reliability in production environments.  


## 🗣 **Discussion Activity**
- 🤖 **How does RAG improve response accuracy compared to traditional LLMs?**  
- ⏳ **What are the biggest challenges in deploying RAG at scale?**  
- ⚖️ **How can organizations balance retrieval speed and accuracy?**  
