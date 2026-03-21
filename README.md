# 🍋 Lemonade Python SDK

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A robust, production-grade Python wrapper for the **Lemonade C++ Backend**. 

This SDK provides a clean, pythonic interface for interacting with local LLMs running on Lemonade. It was built to power **Sorana** (a visual workspace for AI), extracting the core integration logic into a standalone, open-source library for the developer community.

## 🚀 Key Features

* **Auto-Discovery:** Automatically scans multiple ports and hosts to find active Lemonade instances.
* **Low-Overhead Architecture:** Designed as a thin, efficient wrapper to leverage Lemonade's C++ performance with minimal Python latency.
* **Health Checks & Recovery:** Built-in utilities to verify server status and handle connection drops.
* **Type-Safe Client:** Full Python type hinting for better developer experience (IDE autocompletion).
* **Model Management:** Simple API to load, unload, and list models dynamically.
* **Embeddings API:** Generate text embeddings for semantic search, RAG, and clustering (FLM & llamacpp backends).

## 📦 Installation

```bash
pip install .
```

Alternatively, you can install it directly from GitHub:

```bash
pip install git+[https://github.com/Tetramatrix/lemonade-python-sdk.git](https://github.com/Tetramatrix/lemonade-python-sdk.git)
```

## ⚡ Quick Start

### 1. Connecting to Lemonade

The SDK automatically handles port discovery, so you don't need to hardcode localhost:8000.

```python
from lemonade_integration.client import LemonadeClient
from lemonade_integration.port_scanner import find_available_lemonade_port

# Auto-discover running instance
port = find_available_lemonade_port()
if port:
    client = LemonadeClient(base_url=f"http://localhost:{port}")
    if client.health_check():
        print(f"Connected to Lemonade on port {port}")
else:
    print("No Lemonade instance found.")
```

### 2. Chat Completion

```python
response = client.chat_completion(
    model="Llama-3-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Write a Hello World in C++"}
    ],
    temperature=0.7
)

print(response['choices'][0]['message']['content'])
```

### 3. Model Management

```python
# List all available models
models = client.list_models()
for m in models:
    print(f"Found model: {m['id']}")

# Load a specific model into memory
client.load_model("Mistral-7B-v0.1")
```
### 4. Embeddings (NEW)

Generate text embeddings for semantic search, RAG pipelines, and clustering.

```python
# List available embedding models (filtered by 'embeddings' label)
embedding_models = client.list_embedding_models()
for model in embedding_models:
    print(f"Embedding model: {model['id']}")

# Generate embeddings for single text
response = client.embeddings(
    input="Hello, world!",
    model="nomic-embed-text-v1-GGUF"
)

embedding_vector = response["data"][0]["embedding"]
print(f"Vector length: {len(embedding_vector)}")

# Generate embeddings for multiple texts
texts = ["Text 1", "Text 2", "Text 3"]
response = client.embeddings(
    input=texts,
    model="nomic-embed-text-v1-GGUF"
)

for item in response["data"]:
    print(f"Text {item['index']}: {len(item['embedding'])} dimensions")
```

**Supported Backends: (Lemonade)**
- ✅ **FLM (FastFlowLM)** - NPU-accelerated on Windows
- ✅ **llamacpp** (.GGUF models) - CPU/GPU
- ❌ ONNX/OGA - Not supported

## 📚 Documentation
* [Lemonade Server Docs](https://lemonade-server.ai/docs/server/server_spec/) - Official Lemonade documentation

 
### 🖼️ Production Showcase: 

This SDK powers **3 real-world production applications**:

[Sorana](https://tetramatrix.github.io/Sorana/) — AI Visual Workspace
* SDK drives semantic AI grouping of files and folders onto a spatial 2D canvas
* SDK handles auto-discovery and connection to local Lemonade instances (zero config)

[Aicono](https://tetramatrix.github.io/Aicono/) — AI Desktop Icon Organizer *(Featured in [CHIP Magazine](https://www.chip.de/downloads/Aicono_186527264.html) 🇩🇪)*
* SDK drives AI inference for grouping and categorizing desktop icons
* Reached millions of readers via [CHIP](https://www.chip.de/downloads/Aicono_186527264.html), one of Germany's largest IT publications

[TabNeuron](https://tetramatrix.github.io/TabNeuron/) — AI-Powered Tab Organizer
* SDK enables local AI inference for grouping and categorizing browser tabs
* Desktop companion app + browser extension, demonstrating SDK viability in lightweight client architectures

## 🛠️ Project Structure

* **client.py:** Main entry point for API interactions (chat, embeddings, model management).
* **port_scanner.py:** Utilities for detecting Lemonade instances across ports (8000-9000).
* **model_discovery.py:** Logic for fetching and parsing model metadata.
* **request_builder.py:** Helper functions to construct compliant payloads (chat, embeddings).
* **utils.py:** Additional utility functions.

## 🤝 Contributing

Contributions are welcome! This project is intended to help the AMD Ryzen AI and Lemonade community build downstream applications faster.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.