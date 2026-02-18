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

### 🖼️ Production Showcase: Sorana
This SDK was extracted from the core engine of Sorana, a professional visual workspace for AI. It demonstrates the SDK's capability to handle complex, real-world requirements on AMD Ryzen AI hardware:

* **Low Latency:** Powers sub-second response times for multi-model chat interfaces.
* **Dynamic Workflows:** Manages the loading and unloading of 20+ different LLMs based on user activity to optimize local NPU/GPU memory.
* **Zero-Config UX:** Uses the built-in port scanner to automatically connect the Sorana frontend to the Lemonade backend without user intervention.

## 🛠️ Project Structure

* **client.py:** Main entry point for API interactions.
* **port_scanner.py:** Utilities for detecting Lemonade instances across ports (8000-9000).
* **model_discovery.py:** Logic for fetching and parsing model metadata.
* **request_builder.py:** Helper functions to construct compliant payloads.

## 🤝 Contributing

Contributions are welcome! This project is intended to help the AMD Ryzen AI and Lemonade community build downstream applications faster.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
