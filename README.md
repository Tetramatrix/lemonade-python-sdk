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
* **Audio API:** Whisper speech-to-text and Kokoro text-to-speech.
* **Reranking API:** Reorder documents by relevance for better RAG results.
* **Image Generation:** Create images from text prompts using Stable Diffusion.
* **WebSocket Streaming:** Real-time audio transcription with VAD.

## 📦 Installation

```bash
pip install .
```

Alternatively, you can install it directly from GitHub:

```bash
pip install git+https://github.com/Tetramatrix/lemonade-python-sdk.git
```

## ⚡ Quick Start

### 1. Connecting to Lemonade

The SDK automatically handles port discovery, so you don't need to hardcode localhost:8000.

```python
from lemonade_sdk import LemonadeClient, find_available_lemonade_port

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

### 5. Audio Transcription (Whisper) - NEW

Transcribe audio files to text using Whisper.

```python
# List available audio models (Whisper + Kokoro)
audio_models = client.list_audio_models()
for model in audio_models:
    print(f"Audio model: {model['id']}")

# Transcribe an audio file
result = client.transcribe_audio(
    file_path="meeting.wav",
    model="Whisper-Tiny",
    language="en",  # Optional: None for auto-detection
    response_format="json"  # Options: "json", "text", "verbose_json"
)

if "error" not in result:
    print(f"Transcription: {result['text']}")
    # Verbose format also includes: duration, language, segments
```

**Supported Models:**
- `Whisper-Tiny` (~39M parameters)
- `Whisper-Base` (~74M parameters)
- `Whisper-Small` (~244M parameters)

**Supported Formats:** WAV, MP3, FLAC, OGG, WebM

**Backend:** whisper.cpp (NPU-accelerated on Windows)

### 6. Text-to-Speech (Kokoro) - NEW

Generate speech from text using Kokoro TTS.

```python
# Generate speech and save to file
client.text_to_speech(
    input_text="Hello, Lemonade can now speak!",
    model="kokoro-v1",
    voice="shimmer",  # Options: shimmer, corey, af_bella, am_adam, etc.
    speed=1.0,  # 0.5 - 2.0
    response_format="mp3",  # Options: mp3, wav, opus, pcm, aac, flac
    output_file="speech.mp3"  # Saves directly to file
)

# Or get audio bytes directly
audio_bytes = client.text_to_speech(
    input_text="Short test!",
    model="kokoro-v1",
    voice="corey",
    response_format="mp3"
)

with open("speech.mp3", "wb") as f:
    f.write(audio_bytes)
```

**Supported Models:**
- `kokoro-v1` (~82M parameters)

**Available Voices:**

| Voice ID | Language | Gender |
|----------|----------|--------|
| `shimmer` | EN | Female |
| `corey` | EN | Male |
| `af_bella`, `af_nicole` | EN-US | Female |
| `am_adam`, `am_michael` | EN-US | Male |
| `bf_emma`, `bf_isabella` | EN-GB | Female |
| `bm_george`, `bm_lewis` | EN-GB | Male |

**Audio Formats:** MP3, WAV, OPUS, PCM, AAC, FLAC

**Backend:** Kokoros (.onnx, CPU)

### 7. Reranking (NEW)

Rerank documents based on relevance to a query.

```python
result = client.rerank(
    query="What is the capital of France?",
    documents=[
        "Berlin is the capital of Germany.",
        "Paris is the capital of France.",
        "London is the capital of the UK."
    ],
    model="bge-reranker-v2-m3-GGUF"
)

# Results sorted by relevance score
for r in result["results"]:
    print(f"Rank {r['index']}: Score={r['relevance_score']:.2f}")
```

**Supported Models:**
- `bge-reranker-v2-m3-GGUF`
- Other BGE reranker models

**Backend:** llamacpp (.GGUF only, not available for FLM or OGA)

### 8. Image Generation (NEW)

Generate images from text prompts using Stable Diffusion.

```python
# Generate and save to file
client.generate_image(
    prompt="A sunset over mountains with lake reflection",
    model="SD-Turbo",
    size="512x512",
    steps=4,  # SD-Turbo needs only 4 steps
    cfg_scale=1.0,
    output_file="sunset.png"
)

# Or get image bytes
image_bytes = client.generate_image(
    prompt="A cute cat",
    model="SD-Turbo"
)
```

**Supported Models:**
- `SD-Turbo` (fast, 4 steps)
- `SDXL-Turbo` (fast, 4 steps)
- `SD-1.5` (standard, 20 steps)
- `SDXL-Base-1.0` (high quality, 20 steps)

**Image Sizes:** 512x512, 1024x1024, or custom

**Backend:** stable-diffusion.cpp

### 9. WebSocket Streaming (NEW)

Real-time audio transcription with Voice Activity Detection (VAD).

```python
from lemonade_sdk import WhisperWebSocketClient

# Create streaming client
stream = client.create_whisper_stream(model="Whisper-Tiny")
stream.connect()

# Set callback for transcriptions
def on_transcript(text):
    print(f"Heard: '{text}'")

stream.on_transcription(on_transcript)

# Stream audio file (PCM16, 16kHz, mono)
for text in stream.stream("audio.pcm"):
    pass  # Callback handles output

# Or stream from microphone (requires pyaudio)
# for text in stream.stream_microphone():
#     print(f"Heard: {text}")

stream.disconnect()
```

**Audio Format:** 16kHz, mono, PCM16 (16-bit)

**Features:**
- Voice Activity Detection (VAD)
- Real-time streaming
- Microphone support (with pyaudio)
- Configurable sensitivity

**Backend:** whisper.cpp (NPU-accelerated on Windows)

## 📚 Documentation

* **[Embeddings API](docs/embeddings_api.md)** - Complete guide for using embeddings
* **[Audio API](docs/audio_api.md)** - Whisper transcription and Kokoro TTS (documentation)
* **[Implementation Plan](docs/AUDIO_IMPLEMENTATION.md)** - Audio API implementation roadmap
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

* **client.py:** Main entry point for API interactions (chat, embeddings, audio, reranking, images, model management).
* **port_scanner.py:** Utilities for detecting Lemonade instances across ports (8000-9000).
* **model_discovery.py:** Logic for fetching and parsing model metadata.
* **request_builder.py:** Helper functions to construct compliant payloads (chat, embeddings, audio, reranking, images).
* **audio_stream.py:** WebSocket client for real-time audio transcription with VAD.
* **utils.py:** Additional utility functions.

## 🤝 Contributing

Contributions are welcome! This project is intended to help the AMD Ryzen AI and Lemonade community build downstream applications faster.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
