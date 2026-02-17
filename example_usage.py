"""
Example of using the new Lemonade integration module
"""

# Import the new Lemonade module
from lemonade_integration import LemonadeClient, discover_lemonade_models, find_available_lemonade_port

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


if __name__ == "__main__":
    # Find the port first
    available_port = find_available_lemonade_port()
    
    example_usage()
    
    # Use the discovered port for model discovery
    if available_port:
        example_model_discovery(port=available_port)
    else:
        example_model_discovery()