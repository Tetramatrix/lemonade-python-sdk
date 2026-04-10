"""
Module for building API requests to the Lemonade server
"""

import requests
import json
from typing import Dict, Any, List, Optional


def build_chat_completion_payload(model: str, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
    """
    Creates the payload for chat completion requests.

    Args:
        model (str): The name of the model to use
        messages (List[Dict[str, str]]): The messages for the conversation
        **kwargs: Additional parameters for the request

    Returns:
        Dict[str, Any]: The finished payload for the request
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": kwargs.get("stream", False),
    }

    # Add optional parameters if they exist
    optional_params = [
        "temperature", "top_p", "top_k", "max_tokens", "stop",
        "presence_penalty", "frequency_penalty", "repetition_penalty"
    ]

    for param in optional_params:
        if param in kwargs and kwargs[param] is not None:
            payload[param] = kwargs[param]

    # Handle special parameters for Lemonade
    if "options" in kwargs:
        payload["options"] = kwargs["options"]

    return payload


def build_model_load_payload(model_name: str, **kwargs) -> Dict[str, Any]:
    """
    Creates the payload for loading a model.

    Args:
        model_name (str): The name of the model to load
        **kwargs: Additional parameters for loading the model

    Returns:
        Dict[str, Any]: The finished payload for loading the model
    """
    payload = {
        "model": model_name
    }

    # Add optional parameters
    for key, value in kwargs.items():
        payload[key] = value

    return payload


def send_request(url: str, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None, session: Optional[requests.Session] = None, method: str = "POST", timeout: int = 30) -> Dict[str, Any]:
    """
    Sends a request to the Lemonade server.

    Args:
        url (str): The target URL for the request
        payload (Dict[str, Any]): The payload to send
        headers (Optional[Dict[str, str]]): Optional headers for the request
        session (Optional[requests.Session]): Optional session for the request
        method (str): HTTP method to use ("POST" or "GET")
        timeout (int): Request timeout in seconds (default: 30)

    Returns:
        Dict[str, Any]: The response from the server
    """
    if headers is None:
        headers = {
            "Content-Type": "application/json"
        }

    # Use either the passed session or create a temporary one
    req_session = session or requests.Session()

    try:
        if method.upper() == "POST":
            response = req_session.post(url, json=payload, headers=headers, timeout=timeout)
        elif method.upper() == "GET":
            response = req_session.get(url, headers=headers, timeout=timeout, params=payload if payload else None)
        else:
            return {"error": f"Unsupported HTTP method: {method}"}
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error in request to {url}: {e}")
        if response is not None:
            print(f"Response body: {response.text}")
        return {"error": f"HTTP Error: {e}"}
    except requests.exceptions.RequestException as e:
        print(f"Error in request to {url}: {e}")
        return {"error": f"Request Error: {e}"}
    except json.JSONDecodeError as e:
        print(f"Error parsing response from {url}: {e}")
        return {"error": f"JSON Decode Error: {e}"}
    finally:
        # If no session was passed, close the temporary one here
        if session is None:
            req_session.close()


def build_embedding_payload(input_text: str, model: str, **kwargs) -> Dict[str, Any]:
    """
    Creates the payload for embedding requests.

    Args:
        input_text (str): The text for the embedding
        model (str): The name of the model to use
        **kwargs: Additional parameters for the request

    Returns:
        Dict[str, Any]: The finished payload for the embedding request
    """
    payload = {
        "input": input_text,
        "model": model
    }

    # Add optional parameters
    for key, value in kwargs.items():
        payload[key] = value

    return payload


def build_transcription_payload(
    file_path: str,
    model: str,
    language: Optional[str] = None,
    response_format: str = "json"
) -> Dict[str, Any]:
    """
    Creates multipart/form-data payload for audio transcription requests.

    Args:
        file_path (str): Path to the audio file to transcribe
        model (str): The name of the Whisper model to use (e.g., "Whisper-Tiny")
        language (Optional[str]): Optional language code (e.g., "en", "de"). Auto-detects if None.
        response_format (str): Response format - "json", "text", or "verbose_json"

    Returns:
        Dict[str, Any]: Dictionary with 'files' and 'data' keys for requests library
    """
    # Open the file for reading
    files = {
        "file": open(file_path, "rb")
    }

    data = {
        "model": model
    }

    if language:
        data["language"] = language

    data["response_format"] = response_format

    return {
        "files": files,
        "data": data
    }


def send_multipart_request(
    url: str,
    files: Dict[str, Any],
    data: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    session: Optional[requests.Session] = None
) -> Dict[str, Any]:
    """
    Sends a multipart/form-data request to the Lemonade server.

    Args:
        url (str): The target URL for the request
        files (Dict[str, Any]): Files to upload
        data (Optional[Dict[str, Any]]): Form data fields
        headers (Optional[Dict[str, str]]): Optional headers for the request
        session (Optional[requests.Session]): Optional session for the request

    Returns:
        Dict[str, Any]: The response from the server
    """
    # Don't set Content-Type header - requests will set it with boundary for multipart
    if headers is None:
        headers = {}

    # Use either the passed session or create a temporary one
    req_session = session or requests.Session()

    try:
        response = req_session.post(url, files=files, data=data, headers=headers, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error in request to {url}: {e}")
        if response is not None:
            print(f"Response body: {response.text}")
        return {"error": f"HTTP Error: {e}"}
    except requests.exceptions.RequestException as e:
        print(f"Error in request to {url}: {e}")
        return {"error": f"Request Error: {e}"}
    except json.JSONDecodeError as e:
        print(f"Error parsing response from {url}: {e}")
        return {"error": f"JSON Decode Error: {e}"}
    finally:
        # Close the file handles in files dict
        for key, file_obj in files.items():
            if hasattr(file_obj, 'close'):
                file_obj.close()

        # If no session was passed, close the temporary one here
        if session is None:
            req_session.close()


def build_speech_payload(
    input_text: str,
    model: str,
    voice: Optional[str] = None,
    speed: float = 1.0,
    response_format: str = "mp3"
) -> Dict[str, Any]:
    """
    Creates the payload for text-to-speech requests (Kokoro TTS).

    Args:
        input_text (str): The text to synthesize to speech
        model (str): The name of the TTS model to use (e.g., "kokoro-v1")
        voice (Optional[str]): Voice ID (e.g., "shimmer", "corey", "af_bella"). Default: None (auto-select).
        speed (float): Speech speed multiplier (0.5 - 2.0). Default: 1.0
        response_format (str): Audio format - "mp3", "wav", "opus", "pcm", "aac", "flac"

    Returns:
        Dict[str, Any]: The finished payload for the TTS request
    """
    payload = {
        "model": model,
        "input": input_text,
        "response_format": response_format,
        "speed": speed
    }

    # Add optional voice parameter
    if voice:
        payload["voice"] = voice

    return payload


def build_reranking_payload(
    query: str,
    documents: List[str],
    model: str
) -> Dict[str, Any]:
    """
    Creates the payload for reranking requests.

    Args:
        query (str): The search query
        documents (List[str]): List of documents to rerank
        model (str): Reranking model name (e.g., "bge-reranker-v2-m3-GGUF")

    Returns:
        Dict[str, Any]: The finished payload for the reranking request
    """
    return {
        "model": model,
        "query": query,
        "documents": documents
    }


def build_image_generation_payload(
    prompt: str,
    model: str,
    size: str = "512x512",
    steps: int = 4,
    cfg_scale: float = 1.0,
    seed: Optional[int] = None,
    response_format: str = "b64_json"
) -> Dict[str, Any]:
    """
    Creates the payload for image generation requests (Stable Diffusion).

    Args:
        prompt (str): Text description of the image to generate
        model (str): Image generation model (e.g., "SD-Turbo", "SDXL-Turbo")
        size (str): Image size (e.g., "512x512", "1024x1024")
        steps (int): Inference steps (default: 4 for SD-Turbo)
        cfg_scale (float): CFG scale (default: 1.0 for SD-Turbo)
        seed (Optional[int]): Random seed for reproducibility
        response_format (str): Response format ("b64_json" or "url")

    Returns:
        Dict[str, Any]: The finished payload for the image generation request
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "size": size,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "response_format": response_format
    }

    # Add optional seed
    if seed is not None:
        payload["seed"] = seed

    return payload