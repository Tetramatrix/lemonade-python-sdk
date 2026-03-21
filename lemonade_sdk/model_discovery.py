"""
Module for discovering and managing Lemonade models
"""

import requests
import json
from typing import List, Dict, Any, Optional
from .port_scanner import find_available_lemonade_port


def discover_lemonade_models(base_url: str = "http://localhost:8000") -> List[Dict[str, Any]]:
    """
    Scans for available models on the Lemonade server.

    Args:
        base_url (str): The base URL of the Lemonade server

    Returns:
        List[Dict[str, Any]]: List of found models with their properties
    """
    # Try to find the correct port if the default port is not available
    if "localhost" in base_url or "127.0.0.1" in base_url:
        # Extract port from URL
        import re
        port_match = re.search(r':(\d+)', base_url)
        if port_match:
            current_port = int(port_match.group(1))
            available_port = find_available_lemonade_port(ports=[current_port])
            if available_port and available_port != current_port:
                # Replace the port in the URL
                base_url = base_url.replace(f':{current_port}', f':{available_port}')
    
    url = f"{base_url}/api/v1/models"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Lemonade returns models under 'data' key
        models = data.get('data', [])
        
        # Format the models to match the expected structure
        formatted_models = []
        for model in models:
            formatted_model = {
                'id': model.get('id', model.get('name', 'unknown')),
                'name': model.get('name', model.get('id', 'unknown')),
                'object': model.get('object', 'model'),
                'created': model.get('created', 0),
                'owned_by': model.get('owned_by', 'unknown'),
                'source': 'external',
                'provider': 'Lemonade',
                'status': 'Available',
                'size_gb': 0,  # Lemonade doesn't provide size info in the API
                'local_path': f"lemonade://{model.get('name', model.get('id', 'unknown'))}",
                'backend': 'lemonade'
            }
            formatted_models.append(formatted_model)
        
        return formatted_models
    except requests.exceptions.RequestException as e:
        print(f"Error retrieving models from {base_url}: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing response from {base_url}: {e}")
        return []


def get_active_model(base_url: str = "http://localhost:8000") -> Optional[str]:
    """
    Retrieves the currently active model from the Lemonade server.

    Args:
        base_url (str): The base URL of the Lemonade server

    Returns:
        Optional[str]: The name of the active model or None
    """
    # Lemonade has no direct endpoint for the active model,
    # so we try various known endpoints
    endpoints_to_try = [
        f"{base_url}/api/v1/current_model",
        f"{base_url}/api/v1/model",
        f"{base_url}/api/v1/status"
    ]

    for endpoint in endpoints_to_try:
        try:
            response = requests.get(endpoint, timeout=5)
            if response.status_code == 200:
                data = response.json()

                # Try various possible field names for the active model
                for field in ['model', 'current_model', 'active_model', 'name']:
                    if field in data:
                        return data[field]

                # If the result is directly a string
                if isinstance(data, str):
                    return data

        except (requests.exceptions.RequestException, json.JSONDecodeError):
            continue

    # As a fallback, try to get the first available model
    available_models = discover_lemonade_models(base_url)
    if available_models:
        return available_models[0]['name']

    return None


def verify_model_availability(model_name: str, base_url: str = "http://localhost:8000") -> bool:
    """
    Checks if a specific model is available on the Lemonade server.

    Args:
        model_name (str): The name of the model to check
        base_url (str): The base URL of the Lemonade server

    Returns:
        bool: True if the model is available, otherwise False
    """
    available_models = discover_lemonade_models(base_url)
    
    for model in available_models:
        if model['name'] == model_name or model['id'] == model_name:
            return True
    
    return False