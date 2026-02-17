"""
Helper functions for Lemonade integration
"""

import json
from typing import List, Dict, Any


def format_messages_for_lemonade(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Formats messages in Lemonade-compatible format.

    Args:
        messages (List[Dict[str, str]]): The messages to format

    Returns:
        List[Dict[str, str]]: The formatted messages
    """
    # Lemonade expects messages in OpenAI-like format
    # Ensure each message has a 'role' and 'content' field
    formatted_messages = []
    for msg in messages:
        formatted_msg = {
            "role": msg.get("role", "user"),
            "content": msg.get("content", "")
        }
        formatted_messages.append(formatted_msg)

    return formatted_messages


def extract_model_info_from_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts relevant model information from the response.

    Args:
        response (Dict[str, Any]): The response from the server

    Returns:
        Dict[str, Any]: Extracted model information
    """
    model_info = {}

    # Try to extract various possible fields
    if "model" in response:
        model_info["model"] = response["model"]

    if "usage" in response:
        model_info["usage"] = response["usage"]

    if "choices" in response:
        model_info["choices"] = response["choices"]

    if "created" in response:
        model_info["created"] = response["created"]

    # For model list responses
    if "data" in response:
        model_info["data"] = response["data"]

    return model_info


def validate_lemonade_response(response: Dict[str, Any]) -> bool:
    """
    Validates whether the response from the Lemonade server has the expected format.

    Args:
        response (Dict[str, Any]): The response from the server

    Returns:
        bool: True if the response is valid, otherwise False
    """
    # Check for basic fields that a valid response should have
    if not isinstance(response, dict):
        return False

    # For chat completion responses, there should be choices
    if "choices" in response:
        if not isinstance(response["choices"], list) or len(response["choices"]) == 0:
            return False

    # For model list responses, there should be data
    if "data" in response:
        if not isinstance(response["data"], list):
            return False

    return True


def sanitize_model_name(model_name: str) -> str:
    """
    Cleans the model name of invalid characters.

    Args:
        model_name (str): The model name to clean

    Returns:
        str: The cleaned model name
    """
    # Remove or replace invalid characters
    sanitized = model_name.strip()
    # Replace spaces with underscores if necessary
    sanitized = sanitized.replace(" ", "_")
    return sanitized


def format_error_message(error: Exception) -> str:
    """
    Formats an error message for output.

    Args:
        error (Exception): The exception that occurred

    Returns:
        str: Formatted error message
    """
    return f"Error: {type(error).__name__} - {str(error)}"