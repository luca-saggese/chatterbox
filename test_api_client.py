"""
Test client for Chatterbox TTS API
Demonstrates how to call the API and save the generated audio.
"""
import requests
import json
from pathlib import Path

# API endpoint
API_URL = "http://localhost:8000"


def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")


def test_list_languages():
    """Test listing supported languages"""
    print("Listing supported languages...")
    response = requests.get(f"{API_URL}/languages")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Supported languages: {list(data['supported_languages'].keys())}\n")


def test_generate_simple(text: str, output_file: str = "test_output_simple.wav"):
    """Test the simple generation endpoint with query parameters"""
    print(f"Testing simple generation with text: '{text[:50]}...'")
    
    params = {
        "text": text,
        "language_id": "it"
    }
    
    response = requests.post(f"{API_URL}/generate-simple", params=params)
    
    if response.status_code == 200:
        # Save the audio file
        with open(output_file, "wb") as f:
            f.write(response.content)
        print(f"✓ Audio saved to {output_file}")
        print(f"  File size: {len(response.content) / 1024:.2f} KB\n")
    else:
        print(f"✗ Error: {response.status_code}")
        print(f"  {response.text}\n")


def test_generate_full(text: str, output_file: str = "test_output_full.wav"):
    """Test the full generation endpoint with all parameters"""
    print(f"Testing full generation with text: '{text[:50]}...'")
    
    # Request payload
    payload = {
        "text": text,
        "language_id": "it",
        "cfg_weight": 0.2,
        "exaggeration": 1.5,
        "temperature": 0.8,
        "repetition_penalty": 1.2,
        "min_p": 0.05,
        "top_p": 0.95,
        "auto_split": True,
        "sentence_pause_ms": 400
    }
    
    response = requests.post(
        f"{API_URL}/generate",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        # Save the audio file
        with open(output_file, "wb") as f:
            f.write(response.content)
        print(f"✓ Audio saved to {output_file}")
        print(f"  File size: {len(response.content) / 1024:.2f} KB\n")
    else:
        print(f"✗ Error: {response.status_code}")
        print(f"  {response.text}\n")


def test_long_text():
    """Test with a longer text passage"""
    print("Testing with long text...")
    
    text = """
    La Costituzione era custodita in una teca di vetro blindato, illuminata da luci soffuse.
    Un gruppetto di visitatori si avvicinò con l'andatura lenta di chi cammina in un museo.
    Si fermarono a una distanza di sicurezza, come se temessero di danneggiare quel documento
    con il solo respiro.
    """
    
    test_generate_full(text, "test_long_output.wav")


def main():
    """Run all tests"""
    print("=" * 60)
    print("Chatterbox TTS API Test Client")
    print("=" * 60 + "\n")
    
    try:
        # Test 1: Health check
        test_health_check()
        
        # Test 2: List languages
        test_list_languages()
        
        # Test 3: Simple generation
        test_generate_simple("Ciao mondo! Come stai?")
        
        # Test 4: Full generation with custom parameters
        test_generate_full(
            "La tecnologia moderna ha trasformato il modo in cui comunichiamo.",
            "test_custom.wav"
        )
        
        # Test 5: Long text
        test_long_text()
        
        print("=" * 60)
        print("All tests completed!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("\n✗ ERROR: Could not connect to API server.")
        print("  Make sure the server is running: python api_server.py")
    except Exception as e:
        print(f"\n✗ ERROR: {e}")


if __name__ == "__main__":
    main()
