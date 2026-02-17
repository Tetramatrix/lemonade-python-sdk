"""
LemonadeClient - Main class for interacting with the Lemonade server
"""

import requests
import json
from typing import Dict, List, Optional, Any
from .request_builder import build_chat_completion_payload, send_request
from .model_discovery import get_active_model


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