# VectorDB — Build a Vector Database from Scratch in Python

A fully working **Vector Database** built from scratch in Python with a web UI.  
Implements **HNSW**, **KD-Tree**, and **Brute Force** search algorithms side-by-side, plus a **RAG pipeline** powered by a local LLM via Ollama.

> Built as an educational project to show how production vector databases like Pinecone, Weaviate, and Chroma actually work under the hood.

---

## What This Project Does

| Feature | Description |
|---|---|
| **3 Search Algorithms** | HNSW, KD-Tree, and Brute Force — run all three and compare speed |
| **3 Distance Metrics** | Cosine similarity, Euclidean distance, Manhattan distance |
| **16D Demo Vectors** | 20 pre-loaded semantic vectors across 4 categories (CS, Math, Food, Sports) |
| **2D PCA Scatter Plot** | Live visualization of semantic space — watch clusters form |
| **Real Document Embedding** | Paste any text → Ollama embeds it with `nomic-embed-text` |
| **RAG Pipeline** | Ask questions about your documents → HNSW retrieves context → local LLM answers |
| **Full REST API** | CRUD endpoints: insert, delete, search, benchmark, and hnsw-info |

---

## How It Works

```text
Your Text
    │
    ▼
Ollama (nomic-embed-text)          ← converts text to a vector embedding
    │
    ▼
HNSW Index (Python)                ← indexes the vector in a multilayer graph
    │
    ▼
Semantic Search                    ← finds nearest neighbors in vector space
    │
    ▼
Ollama (llama3.2)                  ← reads retrieved chunks, generates an answer
    │
    ▼
Answer
```

**HNSW (Hierarchical Navigable Small World)** is the same algorithm used by Pinecone, Weaviate, Chroma, and Milvus. It builds a multilayer graph where each layer is progressively sparser — searches start at the top layer and zoom in, achieving O(log N) complexity instead of O(N) for brute force.

---

## Prerequisites

You need **3 things** installed on your machine:

1. **Python 3.10+**
2. **Git**
3. **Ollama** (runs the local AI models)

---

## Step-by-Step Setup

### Step 1 — Install Python

1. Download Python from **https://www.python.org/downloads/**
2. Make sure you install Python from the official installer and check **Add Python to PATH**
3. Verify in PowerShell:

```powershell
python --version
```

---

### Step 2 — Install Git

1. Go to **https://git-scm.com/download/win** and download Git for Windows
2. Run the installer with default settings
3. Verify in PowerShell:

```powershell
git --version
```

---

### Step 3 — Install Ollama (Local AI Models)

1. Go to **https://ollama.com** and click **Download for Windows**
2. Run the installer
3. Ollama starts automatically in the system tray
4. Open **PowerShell** and pull the required models:

```powershell
ollama pull nomic-embed-text
```

```powershell
ollama pull llama3.2
```

5. Verify Ollama is running:

```powershell
ollama list
```

> **Minimum specs for Ollama:** 8GB RAM recommended. The models will use roughly 3GB total.

---

### Step 4 — Clone the Repository

Open **PowerShell** and run:

```powershell
git clone https://github.com/YOUR_USERNAME/VectorDB.git
cd VectorDB
```

---

### Step 5 — Install Python Dependencies

Inside the project folder, run:

```powershell
python -m pip install -r requirements.txt
```

If you do not have a `requirements.txt`, install the needed packages manually:

```powershell
python -m pip install flask requests
```

---

### Step 6 — Run the Server

**Terminal 1** — Start Ollama (if it is not already running):

```powershell
ollama serve
```

**Terminal 2** — Start the Python server:

```powershell
python main.py
```

You should see output like:

```text
=== VectorDB Engine ===
http://localhost:8080
20 demo vectors | 16 dims | HNSW+KD-Tree+BruteForce
Ollama: ONLINE
  embed model: nomic-embed-text  gen model: llama3.2
```

Open your browser and go to:

```text
http://localhost:8080
```

---

## Using the Application

### Tab 1: Search (Demo Vectors)

- Type any concept in the search box: `binary tree`, `sushi`, `basketball`, or `calculus`
- Choose your algorithm: **HNSW**, **KD-Tree**, or **Brute Force**
- Choose distance metric: **Cosine**, **Euclidean**, or **Manhattan**
- Click **⚡ SEARCH** — results appear with distances, and the matching point glows on the scatter plot
- Click **▶ COMPARE ALL ALGOS** to run all 3 algorithms and compare their speed

### Tab 2: Documents (Real Embeddings)

This uses Ollama to generate embeddings from any text.

1. Type a title (for example, `Operating Systems Notes`)
2. Paste any text — lecture notes, textbook paragraphs, or Wikipedia articles
3. Click **⚡ EMBED & INSERT**
4. Long documents are automatically split into overlapping 250-word chunks
5. Each chunk gets its own embedding and is stored in a separate HNSW index

### Tab 3: Ask AI (RAG Pipeline)

1. Make sure you have inserted some documents in Tab 2 first
2. Type a question about your documents
3. Click **🤖 ASK AI**

What happens behind the scenes:

```text
1. Your question → embedded with nomic-embed-text (768D vector)
2. HNSW search → finds the most semantically similar chunks
3. Retrieved chunks → sent as context to llama3.2
4. Local LLM → answers the question naturally
```

---

## API Endpoints

This project exposes a small REST API, including:

- `GET /search` — search demo vectors
- `POST /insert` — insert a new demo vector
- `DELETE /delete/<id>` — remove a demo vector
- `GET /benchmark` — compare algorithms
- `GET /hnsw-info` — inspect HNSW graph metadata
- `POST /doc/insert` — insert document chunks
- `POST /doc/ask` — full RAG pipeline
- `GET /status` — check Ollama availability and counts

---

## Notes

This Python version preserves the same learning goals as the original C++ implementation while making the project easier to run, modify, and extend for experimentation.
