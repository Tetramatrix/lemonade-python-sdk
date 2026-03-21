"""
Example of using the new Lemonade integration module
"""

# Import the new Lemonade module
from lemonade_sdk import LemonadeClient, discover_lemonade_models, find_available_lemonade_port

def example_usage():
    """
    Shows how the Lemonade module can be used
    """
    print("=== Lemonade Integration Example ===")

    # Find available Lemonade servers
    available_port = find_available_lemonade_port()
    if available_port:
        print(f"Lemonade server found on port: {available_port}")
        base_url = f"http://localhost:{available_port}"
    else:
        print("No Lemonade server found, using default URL")
        base_url = "http://localhost:8000"

    # Create a Lemonade client
    client = LemonadeClient(base_url=base_url)

    # Check if the server is reachable
    if client.health_check():
        print("✅ Lemonade server is reachable")

        # List available models
        models = client.list_models()
        print(f"Found models: {len(models)}")
        for model in models:
            print(f"  - {model.get('id', model.get('name', 'Unknown'))}")

        # Try a chat request (only if models are available)
        if models:
            sample_model = models[0].get('id', models[0].get('name'))
            print(f"\nTrying chat request with model: {sample_model}")

            response = client.chat_completion(
                model=sample_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello, how are you?"}
                ]
            )

            print(f"Response: {response}")
    else:
        print("❌ Lemonade server is not reachable")


def example_model_discovery(port=8000):
    """
    Shows how model discovery works
    """
    print(f"\n=== Model Discovery Example on port {port} ===")

    # Discover models on the specified server
    models = discover_lemonade_models(f"http://localhost:{port}")
    print(f"Discovered models: {len(models)}")

    for model in models:
        print(f"  - ID: {model['id']}, Name: {model['name']}, Status: {model['status']}")


def example_embeddings(port=None):
    """
    Shows how to use the embeddings API
    """
    print("\n=== Embeddings Example ===")

    # Find available port
    if port is None:
        port = find_available_lemonade_port()

    if not port:
        print("No Lemonade server found!")
        return

    client = LemonadeClient(base_url=f"http://localhost:{port}")

    # Health check
    if not client.health_check():
        print("Lemonade server is not reachable!")
        return

    # List embedding models
    embedding_models = client.list_embedding_models()

    if not embedding_models:
        print("No embedding models available!")
        print("Tip: Load an embedding model like 'nomic-embed-text-v1-GGUF'")
        return

    print(f"Found {len(embedding_models)} embedding model(s):")
    for model in embedding_models:
        print(f"  - {model['id']}")

    # Use the first available embedding model
    model_name = embedding_models[0]['id']
    print(f"\nUsing model: {model_name}")

    # Generate embeddings for single text
    print("\n--- Single Text Embedding ---")
    response = client.embeddings(
        input="Hello, world!",
        model=model_name
    )

    if "error" not in response:
        embedding = response["data"][0]["embedding"]
        print(f"Embedding vector length: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")

    # Generate embeddings for multiple texts
    print("\n--- Multiple Texts Embedding ---")
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science."
    ]

    response = client.embeddings(
        input=texts,
        model=model_name
    )

    if "error" not in response:
        print(f"Generated embeddings for {len(response['data'])} texts:")
        for item in response["data"]:
            print(f"  Text {item['index']}: Vector length = {len(item['embedding'])}")

        # Show token usage
        print(f"\nToken usage:")
        print(f"  Prompt tokens: {response['usage']['prompt_tokens']}")
        print(f"  Total tokens: {response['usage']['total_tokens']}")


def example_whisper_transcription(port=None):
    """
    Shows how to transcribe audio files using Whisper
    """
    print("\n=== Whisper Audio Transcription Example ===")

    # Find available port
    if port is None:
        port = find_available_lemonade_port()

    if not port:
        print("No Lemonade server found!")
        return

    client = LemonadeClient(base_url=f"http://localhost:{port}")

    # Health check
    if not client.health_check():
        print("Lemonade server is not reachable!")
        return

    # List audio models
    audio_models = client.list_audio_models()

    if not audio_models:
        print("No audio models available!")
        print("Tip: Load a Whisper model like 'Whisper-Tiny'")
        return

    print(f"Found {len(audio_models)} audio model(s):")
    for model in audio_models:
        print(f"  - {model['id']}")

    # Filter for Whisper models
    whisper_models = [m for m in audio_models if "whisper" in m["id"].lower()]

    if not whisper_models:
        print("\nNo Whisper models found!")
        return

    # Use the first Whisper model
    model_name = whisper_models[0]["id"]
    print(f"\nUsing Whisper model: {model_name}")

    # Note: This example requires an actual audio file
    print("\n--- Audio Transcription ---")
    print("Note: To test transcription, place an audio file (e.g., 'test_audio.wav') in the current directory")

    # Try to transcribe a test file
    test_file = "test_audio.wav"
    import os
    if os.path.exists(test_file):
        print(f"Transcribing: {test_file}")

        response = client.transcribe_audio(
            file_path=test_file,
            model=model_name,
            language="en",  # Optional: set to None for auto-detection
            response_format="json"
        )

        if "error" not in response:
            print(f"\n✅ Transcription successful!")
            print(f"Text: {response.get('text', 'N/A')}")

            # Show verbose info if available
            if "duration" in response:
                print(f"Duration: {response['duration']:.2f}s")
            if "language" in response:
                print(f"Language: {response['language']}")
        else:
            print(f"\n❌ Error: {response.get('error', 'Unknown error')}")
    else:
        print(f"\n⚠️ Test file '{test_file}' not found.")
        print("To test transcription:")
        print(f"  1. Place an audio file at: {os.path.abspath(test_file)}")
        print(f"  2. Run this example again")
        print("\nSupported formats: WAV, MP3, FLAC, OGG, WebM")


def example_kokoro_tts(port=None):
    """
    Shows how to generate speech using Kokoro TTS
    """
    print("\n=== Kokoro Text-to-Speech Example ===")

    # Find available port
    if port is None:
        port = find_available_lemonade_port()

    if not port:
        print("No Lemonade server found!")
        return

    client = LemonadeClient(base_url=f"http://localhost:{port}")

    # Health check
    if not client.health_check():
        print("Lemonade server is not reachable!")
        return

    # List audio models
    audio_models = client.list_audio_models()

    if not audio_models:
        print("No audio models available!")
        print("Tip: Load a Kokoro model like 'kokoro-v1'")
        return

    print(f"Found {len(audio_models)} audio model(s):")
    for model in audio_models:
        print(f"  - {model['id']}")

    # Filter for Kokoro models
    kokoro_models = [m for m in audio_models if "kokoro" in m["id"].lower()]

    if not kokoro_models:
        print("\n⚠️ No Kokoro model found!")
        print("Tip: Load kokoro-v1 model for text-to-speech")
        return

    # Use the first Kokoro model
    model_name = kokoro_models[0]["id"]
    print(f"\nUsing Kokoro model: {model_name}")

    # Generate speech
    text = "Hello, Lemonade can now speak!"
    print(f"\nSynthesizing: '{text}'")

    # Generate and save to file
    output_file = "output_speech.mp3"
    print(f"Saving to: {output_file}")

    result = client.text_to_speech(
        input_text=text,
        model=model_name,
        voice="shimmer",  # Available: shimmer, corey, af_bella, am_adam, etc.
        response_format="mp3",
        output_file=output_file
    )

    if result is None:
        print(f"\n✅ Speech generated successfully!")
        print(f"Saved to: {output_file}")

        # Also demonstrate getting bytes directly
        print("\n--- Getting Audio Bytes Directly ---")
        audio_bytes = client.text_to_speech(
            input_text="Short test!",
            model=model_name,
            voice="corey",
            response_format="mp3"
        )

        if isinstance(audio_bytes, bytes):
            print(f"Received {len(audio_bytes)} bytes of audio data")
        elif isinstance(audio_bytes, dict) and "error" in audio_bytes:
            print(f"Error: {audio_bytes['error']}")
    elif isinstance(result, dict) and "error" in result:
        print(f"\n❌ Error: {result['error']}")


def example_reranking(port=None):
    """
    Shows how to rerank documents based on relevance to a query
    """
    print("\n=== Reranking Example ===")

    # Find available port
    if port is None:
        port = find_available_lemonade_port()

    if not port:
        print("No Lemonade server found!")
        return

    client = LemonadeClient(base_url=f"http://localhost:{port}")

    # Health check
    if not client.health_check():
        print("Lemonade server is not reachable!")
        return

    # List all models to find reranking models
    all_models = client.list_models()
    rerank_models = [m for m in all_models if "rerank" in m.get("id", "").lower() or "reranking" in m.get("labels", [])]

    if not rerank_models:
        print("\n⚠️ No reranking model found!")
        print("Tip: Load a reranking model like 'bge-reranker-v2-m3-GGUF'")
        return

    print(f"Found {len(rerank_models)} reranking model(s):")
    for model in rerank_models:
        print(f"  - {model['id']}")

    # Use the first reranking model
    model_name = rerank_models[0]["id"]
    print(f"\nUsing model: {model_name}")

    # Example query and documents
    query = "What is the capital of France?"
    documents = [
        "Berlin is the capital and largest city of Germany.",
        "Paris is the capital and most populous city of France.",
        "London is the capital of the United Kingdom.",
        "The Eiffel Tower is located in Paris.",
        "French cuisine is renowned worldwide."
    ]

    print(f"\nQuery: '{query}'")
    print(f"\nDocuments ({len(documents)}):")
    for i, doc in enumerate(documents):
        print(f"  {i}. {doc[:60]}...")

    # Rerank documents
    print("\n--- Reranking ---")
    result = client.rerank(
        query=query,
        documents=documents,
        model=model_name
    )

    if "error" not in result:
        print("\n✅ Reranking successful!")
        print("\nResults (sorted by relevance):")

        results = result.get("results", [])
        for r in results:
            idx = r.get("index", 0)
            score = r.get("relevance_score", 0)
            print(f"  Rank {r.get('index', 0)}: Score={score:.2f} - '{documents[idx][:50]}...'")
    else:
        print(f"\n❌ Error: {result.get('error', 'Unknown error')}")


def example_image_generation(port=None):
    """
    Shows how to generate images using Stable Diffusion
    """
    print("\n=== Image Generation Example ===")

    # Find available port
    if port is None:
        port = find_available_lemonade_port()

    if not port:
        print("No Lemonade server found!")
        return

    client = LemonadeClient(base_url=f"http://localhost:{port}")

    # Health check
    if not client.health_check():
        print("Lemonade server is not reachable!")
        return

    # List all models to find image generation models
    all_models = client.list_models()
    image_models = [m for m in all_models if any(x in m.get("id", "").lower() for x in ["sd-", "stable-diffusion", "sdxl"])]

    if not image_models:
        print("\n⚠️ No image generation model found!")
        print("Tip: Load an image model like 'SD-Turbo'")
        return

    print(f"Found {len(image_models)} image generation model(s):")
    for model in image_models:
        print(f"  - {model['id']}")

    # Use the first image model
    model_name = image_models[0]["id"]
    print(f"\nUsing model: {model_name}")

    # Generate image
    prompt = "A beautiful sunset over mountains with a lake reflection"
    output_file = "generated_sunset.png"

    print(f"\nGenerating: '{prompt}'")
    print(f"Saving to: {output_file}")

    result = client.generate_image(
        prompt=prompt,
        model=model_name,
        size="512x512",
        steps=4,  # SD-Turbo needs only 4 steps
        cfg_scale=1.0,
        output_file=output_file
    )

    if result is None:
        print(f"\n✅ Image generated successfully!")
        print(f"Saved to: {output_file}")
    elif isinstance(result, dict) and "error" in result:
        print(f"\n❌ Error: {result['error']}")


def example_whisper_websocket(port=None):
    """
    Shows real-time transcription via WebSocket
    """
    print("\n=== Whisper WebSocket Streaming Example ===")

    # Find available port
    if port is None:
        port = find_available_lemonade_port()

    if not port:
        print("No Lemonade server found!")
        return

    client = LemonadeClient(base_url=f"http://localhost:{port}")

    # Health check
    if not client.health_check():
        print("Lemonade server is not reachable!")
        return

    # Check for Whisper model
    audio_models = client.list_audio_models()
    whisper_models = [m for m in audio_models if "whisper" in m["id"].lower()]

    if not whisper_models:
        print("\n⚠️ No Whisper model found!")
        print("Tip: Load a Whisper model like 'Whisper-Tiny'")
        return

    model_name = whisper_models[0]["id"]
    print(f"Using Whisper model: {model_name}")

    # Create streaming client
    stream = client.create_whisper_stream(model=model_name)

    # Test file for streaming
    test_file = "test_audio.pcm"

    print(f"\n--- WebSocket Streaming ---")
    print(f"Note: This example requires a PCM audio file at: {test_file}")
    print("Format: 16kHz, mono, PCM16 (16-bit)")

    import os
    if os.path.exists(test_file):
        print(f"\nStreaming: {test_file}")

        try:
            stream.connect()
            print("WebSocket connected!")

            # Set callback for transcriptions
            def on_transcript(text):
                print(f"  🎤 Heard: '{text}'")

            stream.on_transcription(on_transcript)

            # Stream the audio file
            print("\nTranscribing...")
            for transcription in stream.stream(test_file):
                pass  # Callback handles output

            print("\n✅ Streaming complete!")

        except Exception as e:
            print(f"\n❌ Error: {e}")
        finally:
            stream.disconnect()
            print("WebSocket disconnected!")
    else:
        print(f"\n⚠️ Test file '{test_file}' not found.")
        print("\nTo test WebSocket streaming:")
        print(f"  1. Convert audio to PCM format:")
        print(f"     ffmpeg -i audio.wav -f s16le -acodec pcm_s16le -ar 16000 -ac 1 {test_file}")
        print(f"  2. Place the file at: {os.path.abspath(test_file)}")
        print(f"  3. Run this example again")
        print("\nOr use microphone streaming (requires pyaudio):")
        print("  pip install pyaudio")
        print("  Then use: stream.stream_microphone()")


if __name__ == "__main__":
    # Find the port first
    available_port = find_available_lemonade_port()

    example_usage()

    # Use the discovered port for model discovery
    if available_port:
        example_model_discovery(port=available_port)
    else:
        example_model_discovery()

    # Run embeddings example
    example_embeddings(port=available_port)

    # Run Whisper transcription example
    example_whisper_transcription(port=available_port)

    # Run Kokoro TTS example
    example_kokoro_tts(port=available_port)

    # Run reranking example
    example_reranking(port=available_port)

    # Run image generation example
    example_image_generation(port=available_port)

    # Run WebSocket streaming example
    example_whisper_websocket(port=available_port)