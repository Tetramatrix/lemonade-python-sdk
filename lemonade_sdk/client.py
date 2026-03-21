"""
LemonadeClient - Main class for interacting with the Lemonade server
"""

import requests
import json
from typing import Dict, List, Optional, Any
from .request_builder import build_chat_completion_payload, send_request, build_embedding_payload, build_transcription_payload, send_multipart_request, build_speech_payload, build_reranking_payload, build_image_generation_payload
from .model_discovery import get_active_model
from .audio_stream import WhisperWebSocketClient


class LemonadeClient:
    """
    A class for interacting with the Lemonade LLM Server.
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initializes the Lemonade Client with the base URL.

        Args:
            base_url (str): The base URL of the Lemonade server (default: http://localhost:8000)
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
    def list_models(self) -> List[Dict[str, Any]]:
        """
        Retrieves the list of available models from the Lemonade server.

        Returns:
            List[Dict[str, Any]]: List of available models
        """
        url = f"{self.base_url}/api/v1/models"
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Lemonade returns models under 'data' key
            models = data.get('data', [])
            return models
        except requests.exceptions.RequestException as e:
            print(f"Error retrieving models: {e}")
            return []
        except json.JSONDecodeError as e:
            print(f"Error parsing response: {e}")
            return []
    
    def chat_completion(self, model: str, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Sends a chat completion request to the Lemonade server.

        Args:
            model (str): The name of the model to use
            messages (List[Dict[str, str]]): The messages for the conversation
            **kwargs: Additional parameters for the request

        Returns:
            Dict[str, Any]: The response from the server
        """
        url = f"{self.base_url}/api/v1/chat/completions"

        payload = build_chat_completion_payload(model, messages, **kwargs)

        try:
            response = send_request(url, payload, session=self.session)
            return response
        except Exception as e:
            print(f"Error in chat completion request: {e}")
            return {"error": str(e)}
    
    def health_check(self) -> bool:
        """
        Checks if the Lemonade server is running and reachable.

        Returns:
            bool: True if the server is reachable, otherwise False
        """
        try:
            models = self.list_models()
            return len(models) >= 0  # If we don't get an error, the server is reachable
        except:
            return False
    
    def get_current_model(self) -> Optional[str]:
        """
        Retrieves the currently active model from the Lemonade server.

        Returns:
            Optional[str]: The name of the current model or None
        """
        return get_active_model(self.base_url)
    
    def load_model(self, model_name: str, **kwargs) -> Dict[str, Any]:
        """
        Loads a specific model on the Lemonade server.

        Args:
            model_name (str): The name of the model to load
            **kwargs: Additional parameters for loading the model

        Returns:
            Dict[str, Any]: Response from the server
        """
        url = f"{self.base_url}/api/v1/load_model"

        payload = {
            "model": model_name,
            **kwargs
        }

        try:
            response = send_request(url, payload, session=self.session)
            return response
        except Exception as e:
            print(f"Error loading model: {e}")
            return {"error": str(e)}
    
    def unload_model(self) -> Dict[str, Any]:
        """
        Unloads the current model from the Lemonade server.

        Returns:
            Dict[str, Any]: Response from the server
        """
        url = f"{self.base_url}/api/v1/unload_model"

        try:
            response = send_request(url, {}, session=self.session)
            return response
        except Exception as e:
            print(f"Error unloading model: {e}")
            return {"error": str(e)}

    def embeddings(self, input: str, model: str, **kwargs) -> Dict[str, Any]:
        """
        Sends an embedding request to the Lemonade server.

        Args:
            input (str): The text or list of texts to embed
            model (str): The name of the embedding model to use (e.g., "nomic-embed-text-v1-GGUF")
            **kwargs: Additional parameters for the request
                - encoding_format (str): "float" (default) or "base64"

        Returns:
            Dict[str, Any]: The response containing the embedding vectors
        """
        url = f"{self.base_url}/api/v1/embeddings"

        payload = build_embedding_payload(input, model, **kwargs)

        try:
            response = send_request(url, payload, session=self.session)
            return response
        except Exception as e:
            print(f"Error in embedding request: {e}")
            return {"error": str(e)}

    def list_embedding_models(self) -> List[Dict[str, Any]]:
        """
        Retrieves only the embedding models from the Lemonade server.
        Filters models by the 'embeddings' label.

        Returns:
            List[Dict[str, Any]]: List of available embedding models
        """
        all_models = self.list_models()

        # Filter models that have the 'embeddings' label
        embedding_models = []
        for model in all_models:
            labels = model.get("labels", [])
            if labels and "embeddings" in labels:
                embedding_models.append(model)

        return embedding_models

    def transcribe_audio(
        self,
        file_path: str,
        model: str,
        language: Optional[str] = None,
        response_format: str = "json"
    ) -> Dict[str, Any]:
        """
        Transcribes an audio file to text using Whisper.

        Args:
            file_path (str): Path to the audio file to transcribe (WAV, MP3, FLAC, etc.)
            model (str): The name of the Whisper model to use (e.g., "Whisper-Tiny", "Whisper-Base")
            language (Optional[str]): Optional language code (e.g., "en", "de"). Auto-detects if None.
            response_format (str): Response format - "json", "text", or "verbose_json"

        Returns:
            Dict[str, Any]: The transcription result. For "json" format: {"text": "transcription"}
        """
        import os
        
        # Validate file exists
        if not os.path.exists(file_path):
            return {"error": f"Audio file not found: {file_path}"}
        
        url = f"{self.base_url}/api/v1/audio/transcriptions"

        payload = build_transcription_payload(file_path, model, language, response_format)

        try:
            response = send_multipart_request(
                url,
                files=payload["files"],
                data=payload["data"],
                session=self.session
            )
            return response
        except Exception as e:
            print(f"Error in audio transcription request: {e}")
            return {"error": str(e)}

    def list_audio_models(self) -> List[Dict[str, Any]]:
        """
        Retrieves audio models (Whisper + Kokoro) from the Lemonade server.
        Filters models by the 'audio' label.

        Returns:
            List[Dict[str, Any]]: List of available audio models
        """
        all_models = self.list_models()

        # Filter models that have the 'audio' label
        audio_models = []
        for model in all_models:
            labels = model.get("labels", [])
            if labels and "audio" in labels:
                audio_models.append(model)

        return audio_models

    def text_to_speech(
        self,
        input_text: str,
        model: str = "kokoro-v1",
        voice: Optional[str] = None,
        speed: float = 1.0,
        response_format: str = "mp3",
        output_file: Optional[str] = None
    ) -> Optional[bytes]:
        """
        Synthesizes speech from text using Kokoro TTS.

        Args:
            input_text (str): Text to synthesize to speech
            model (str): TTS model name (default: "kokoro-v1")
            voice (Optional[str]): Voice ID (e.g., "shimmer", "corey", "af_bella", "am_adam")
            speed (float): Speed multiplier (0.5 - 2.0, default: 1.0)
            response_format (str): Audio format ("mp3", "wav", "opus", "pcm", "aac", "flac")
            output_file (Optional[str]): If provided, saves audio to this file path

        Returns:
            Optional[bytes]: Audio file bytes if output_file is None, otherwise None
        """
        url = f"{self.base_url}/api/v1/audio/speech"

        payload = build_speech_payload(input_text, model, voice, speed, response_format)

        try:
            response = self.session.post(url, json=payload, timeout=60)
            response.raise_for_status()

            # Save to file if specified
            if output_file:
                with open(output_file, "wb") as f:
                    f.write(response.content)
                return None
            else:
                return response.content

        except requests.exceptions.HTTPError as e:
            print(f"HTTP error in TTS request: {e}")
            if response is not None:
                print(f"Response body: {response.text}")
            return {"error": f"HTTP Error: {e}"}
        except requests.exceptions.RequestException as e:
            print(f"Error in TTS request: {e}")
            return {"error": f"Request Error: {e}"}

    def create_whisper_stream(self, model: str = "Whisper-Tiny") -> WhisperWebSocketClient:
        """
        Creates a WebSocket client for real-time audio transcription.

        Args:
            model (str): Whisper model to use (default: "Whisper-Tiny")

        Returns:
            WhisperWebSocketClient: Streaming transcription client

        Example:
            ```python
            stream = client.create_whisper_stream(model="Whisper-Tiny")
            stream.connect()

            for text in stream.stream("audio.pcm"):
                print(f"Heard: {text}")

            stream.disconnect()
            ```
        """
        # Extract websocket port from base_url (default: 9000)
        ws_port = 9000
        try:
            # Try to extract port from base_url
            if ":" in self.base_url:
                port_str = self.base_url.split(":")[-1].rstrip("/")
                if port_str.isdigit():
                    ws_port = int(port_str) + 1  # WebSocket port is typically HTTP port + 1
        except:
            pass

        ws_url = f"ws://localhost:{ws_port}"
        return WhisperWebSocketClient(base_url=ws_url)

    def rerank(
        self,
        query: str,
        documents: List[str],
        model: str
    ) -> Dict[str, Any]:
        """
        Reranks documents based on relevance to a query.

        Args:
            query (str): The search query
            documents (List[str]): List of documents to rerank
            model (str): Reranking model name (e.g., "bge-reranker-v2-m3-GGUF")

        Returns:
            Dict[str, Any]: Reranking results with relevance scores

        Example:
            ```python
            result = client.rerank(
                query="What is the capital of France?",
                documents=["Paris is beautiful", "Berlin is the capital of Germany"],
                model="bge-reranker-v2-m3-GGUF"
            )
            # Returns documents sorted by relevance
            ```
        """
        url = f"{self.base_url}/api/v1/reranking"

        payload = build_reranking_payload(query, documents, model)

        try:
            response = send_request(url, payload, session=self.session)
            return response
        except Exception as e:
            print(f"Error in reranking request: {e}")
            return {"error": str(e)}

    def generate_image(
        self,
        prompt: str,
        model: str = "SD-Turbo",
        size: str = "512x512",
        steps: int = 4,
        cfg_scale: float = 1.0,
        seed: Optional[int] = None,
        response_format: str = "b64_json",
        output_file: Optional[str] = None
    ) -> Optional[bytes]:
        """
        Generates an image from a text prompt using Stable Diffusion.

        Args:
            prompt (str): Text description of the image to generate
            model (str): Image generation model (default: "SD-Turbo")
            size (str): Image size (default: "512x512", options: "512x512", "1024x1024")
            steps (int): Inference steps (default: 4 for SD-Turbo, 20 for others)
            cfg_scale (float): CFG scale (default: 1.0 for SD-Turbo, 7.0 for others)
            seed (Optional[int]): Random seed for reproducibility
            response_format (str): Response format ("b64_json" or "url")
            output_file (Optional[str]): If provided, saves image to this file path

        Returns:
            Optional[bytes]: Image bytes if output_file is None and format is b64_json,
                           otherwise None (saved to file) or dict with error

        Example:
            ```python
            # Generate and save to file
            client.generate_image(
                prompt="A sunset over mountains",
                model="SD-Turbo",
                output_file="sunset.png"
            )

            # Get image bytes
            image_bytes = client.generate_image(
                prompt="A cute cat",
                model="SD-Turbo"
            )
            ```
        """
        url = f"{self.base_url}/api/v1/images/generations"

        payload = build_image_generation_payload(
            prompt, model, size, steps, cfg_scale, seed, response_format
        )

        try:
            response = self.session.post(url, json=payload, timeout=120)
            response.raise_for_status()

            data = response.json()

            # Extract image data
            if response_format == "b64_json":
                import base64
                image_data = data.get("data", [{}])[0].get("b64_json", "")

                if not image_data:
                    return {"error": "No image data in response"}

                image_bytes = base64.b64decode(image_data)

                if output_file:
                    with open(output_file, "wb") as f:
                        f.write(image_bytes)
                    return None
                else:
                    return image_bytes
            else:
                # URL format
                image_url = data.get("data", [{}])[0].get("url", "")
                if output_file and image_url:
                    # Download and save
                    img_response = requests.get(image_url)
                    with open(output_file, "wb") as f:
                        f.write(img_response.content)
                    return None
                return data

        except requests.exceptions.HTTPError as e:
            print(f"HTTP error in image generation: {e}")
            if response is not None:
                print(f"Response body: {response.text}")
            return {"error": f"HTTP Error: {e}"}
        except requests.exceptions.RequestException as e:
            print(f"Error in image generation: {e}")
            return {"error": f"Request Error: {e}"}
        except Exception as e:
            print(f"Error processing image response: {e}")
            return {"error": str(e)}