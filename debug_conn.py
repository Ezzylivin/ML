import requests
import os

# Check if an environment variable is overriding the default
env_url = os.getenv("ML_SERVER_URL")
print(f"DEBUG: ML_SERVER_URL is currently: {env_url}")

target_url = (env_url or "http://127.0.0.1:8000") + "/api/ml/available-models"
print(f"DEBUG: Attempting to connect to: {target_url}")

try:
    # 5-second timeout, plain request without retry logic first
    response = requests.get(target_url, timeout=5)
    print(f"DEBUG: Status Code: {response.status_code}")
    print(f"DEBUG: Response Content: {response.text[:100]}...")
except Exception as e:
    print("\n!!! CONNECTION FAILED !!!")
    print(f"Error Type: {type(e).__name__}")
    print(f"Error Message: {e}")
    
    # Check for common proxy issues
    if "ProxyError" in str(type(e).__name__):
        print("\nTIP: You might have HTTP_PROXY set in your environment variables.")
