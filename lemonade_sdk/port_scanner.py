"""
Module for scanning available Lemonade servers
"""

import socket
import requests
from typing import List, Optional

# Default ports for Lemonade server discovery
# Lemonade typically runs on every 20th port in the 8000-9000 range
LEMONADE_DEFAULT_PORTS = [8000, 8020, 8040, 8060, 8080, 9000, 13305, 11434]  # Standard ports + new defaults (13305, 11434)


def is_port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    """
    Checks if a TCP port is reachable on the host.

    Args:
        host (str): The host on which to check the port
        port (int): The port to check
        timeout (float): Timeout for the connection (default: 0.5 seconds)

    Returns:
        bool: True if the port is reachable, otherwise False
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def verify_lemonade_server(port: int, host: str = "127.0.0.1") -> bool:
    """
    Checks if a Lemonade server is running on the specified port.

    Distinguishes between real Ollama and Lemonade's Ollama-compatible endpoint
    by checking the native Lemonade API first, then falling back to Server headers.

    Args:
        port (int): The port to check
        host (str): The host (default: 127.0.0.1)

    Returns:
        bool: True if a Lemonade server is running on the port, otherwise False
    """
    # Check 1: Native Lemonade API (definitive proof it's Lemonade, not Ollama)
    url = f"http://{host}:{port}/api/v1/models"

    try:
        response = requests.get(url, timeout=2)
        # A Lemonade server should respond to this endpoint with a list of models
        if response.status_code == 200:
            try:
                data = response.json()
                # Check if the response has the expected format
                if "data" in data or isinstance(data, list):
                    return True
            except ValueError:
                # If the response is not valid JSON, it's probably not a Lemonade server
                pass
    except requests.RequestException:
        pass

    # Check 2: For Ollama-compatible port (11434), verify it's Lemonade and not real Ollama
    # Lemonade may run on port 11434 with Ollama API simulation
    # Distinguish by checking Server header or alternative endpoints
    if port == 11434:
        return _is_lemonade_on_ollama_port(host, port)

    return False


def _is_lemonade_on_ollama_port(host: str = "127.0.0.1", port: int = 11434) -> bool:
    """
    Determines if a server on the Ollama-compatible port (11434) is actually Lemonade.

    Real Ollama and Lemonade both respond to Ollama API endpoints on port 11434.
    This function uses multiple detection methods to distinguish between them.

    Args:
        host (str): The host to check (default: 127.0.0.1)
        port (int): The port to check (default: 11434)

    Returns:
        bool: True if it's Lemonade, False if it's real Ollama or no server
    """
    # Method 1: Check Server header in HTTP response
    url = f"http://{host}:{port}/api/tags"  # Ollama-compatible endpoint

    try:
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            # Check Server header for "lemonade" identifier
            server_header = response.headers.get("Server", "").lower()
            if "lemonade" in server_header:
                return True

            # Method 2: Check custom headers that Lemonade might add
            # Some Lemonade versions add X-Server or similar headers
            for header_name in response.headers:
                header_value = response.headers[header_name].lower()
                if "lemonade" in header_name.lower() or "lemonade" in header_value:
                    return True
    except requests.RequestException:
        pass

    # Method 3: Check if native Lemonade API is also available on a nearby port
    # Lemonade often runs both native API (8000-9000) and Ollama compat (11434) together
    # This is a heuristic - not definitive but useful as a fallback
    # We don't scan here to avoid performance impact, just return False
    # The main scanner will find Lemonade on native ports anyway

    return False


def find_available_lemonade_port(
    host: str = "127.0.0.1",
    ports: List[int] = None
) -> Optional[int]:
    """
    Finds the first available port on which a Lemonade server is running.

    Args:
        host (str): The host to search on (default: 127.0.0.1)
        ports (List[int]): List of ports to check (default: [8000, 8020, 8040, 8060, 8080, 9000, 13305, 11434])

    Returns:
        Optional[int]: The first found port with a Lemonade server, or None
    """
    if ports is None:
        ports = LEMONADE_DEFAULT_PORTS

    for port in ports:
        if is_port_open(host, port):
            if verify_lemonade_server(port, host):
                return port

    return None


def scan_multiple_hosts_for_lemonade(
    hosts: List[str] = None,
    ports: List[int] = None
) -> List[tuple]:
    """
    Scans multiple hosts and ports for available Lemonade servers.

    Args:
        hosts (List[str]): List of hosts to scan
        ports (List[int]): List of ports to scan (default: [8000, 8020, 8040, 8060, 8080, 9000, 13305, 11434])

    Returns:
        List[tuple]: List of (host, port) tuples where Lemonade servers were found
    """
    if hosts is None:
        hosts = ["127.0.0.1", "localhost"]

    if ports is None:
        ports = LEMONADE_DEFAULT_PORTS

    available_servers = []

    for host in hosts:
        for port in ports:
            if is_port_open(host, port):
                if verify_lemonade_server(port, host):
                    available_servers.append((host, port))

    return available_servers