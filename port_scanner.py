"""
Module for scanning available Lemonade servers
"""

import socket
import requests
from typing import List, Optional


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

    Args:
        port (int): The port to check
        host (str): The host (default: 127.0.0.1)

    Returns:
        bool: True if a Lemonade server is running on the port, otherwise False
    """
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

    return False


def find_available_lemonade_port(
    host: str = "127.0.0.1",
    ports: List[int] = None
) -> Optional[int]:
    """
    Finds the first available port on which a Lemonade server is running.

    Args:
        host (str): The host to search on (default: 127.0.0.1)
        ports (List[int]): List of ports to check (default: [8000, 8020, 8040, 8060, 8080, 9000])

    Returns:
        Optional[int]: The first found port with a Lemonade server, or None
    """
    if ports is None:
        # Standard Lemonade ports
        ports = [8000, 8020, 8040, 8060, 8080, 9000]
    
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
        ports (List[int]): List of ports to scan

    Returns:
        List[tuple]: List of (host, port) tuples where Lemonade servers were found
    """
    if hosts is None:
        hosts = ["127.0.0.1", "localhost"]

    if ports is None:
        ports = [8000, 8020, 8040, 8060, 8080, 9000]
    
    available_servers = []
    
    for host in hosts:
        for port in ports:
            if is_port_open(host, port):
                if verify_lemonade_server(port, host):
                    available_servers.append((host, port))
    
    return available_servers