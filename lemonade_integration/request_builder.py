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


def send_request(url: str, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None, session: Optional[requests.Session] = None) -> Dict[str, Any]:
    """
    Sends a request to the Lemonade server.

    Args:
        url (str): The target URL for the request
        payload (Dict[str, Any]): The payload to send
        headers (Optional[Dict[str, str]]): Optional headers for the request
        session (Optional[requests.Session]): Optional session for the request

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
        response = req_session.post(url, json=payload, headers=headers, timeout=30)
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