import json
import requests
import sys
from pathlib import Path

def main():
    payload_path = Path("data/sample_scan.json")
    if not payload_path.exists():
        print(f"Error: Could not find payload at {payload_path}")
        sys.exit(1)
        
    with open(payload_path, "r") as f:
        payload = json.load(f)
        
    print("Sending POST request to http://127.0.0.1:8000/analyze...")
    try:
        response = requests.post("http://127.0.0.1:8000/analyze", json=payload)
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the server. Is it running on http://127.0.0.1:8000?")
        sys.exit(1)
        
    print(f"\nResponse Status Code: {response.status_code}\n")
    
    try:
        json_resp = response.json()
        print(json.dumps(json_resp, indent=2))
    except json.JSONDecodeError:
        print("Response was not JSON:")
        print(response.text)
        
if __name__ == "__main__":
    main()
