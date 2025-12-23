import requests
import os
import time

# Configuration
BASE_URL = "http://localhost:3001"
TEST_EMAIL = "test_mobile_user_v1@example.com"
TEST_PASSWORD = "password123"
TEST_NAME = "Test Mobile User"

# Use the image the user uploaded recently as a test image
IMAGE_PATH = r"C:/Users/chris/.gemini/antigravity/brain/50ad9c5d-c1a2-484a-8318-ff6e75f96497/uploaded_image_1766305876789.png"

# Colors for output
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

def log(msg, success=True):
    color = GREEN if success else RED
    print(f"{color}[TEST] {msg}{RESET}")

def print_debug_routes():
    log("Fetching DEBUG ROUTES...", True)
    try:
        resp = requests.get(f"{BASE_URL}/api/debug/routes")
        if resp.status_code == 200:
            routes = resp.json().get("routes", [])
            found = False
            for r in routes:
                if "/api/auth/login" in r["path"]:
                    log(f"FOUND ROUTE: {r['path']} methods={r['methods']}", True)
                    found = True
            if not found:
                 log("CRITICAL: /api/auth/login NOT FOUND in routes!", False)
                 print(routes) # Print all for inspection
        else:
            log(f"Debug Routes Failed: {resp.status_code}", False)
    except Exception as e:
        log(f"Debug Routes Error: {e}", False)

def test_signup_flow():
    log("Testing Signup Flow...", True)
    
    # 1. Signup Init (OTP)
    try:
        resp = requests.post(f"{BASE_URL}/api/auth/signup-init", json={"email": TEST_EMAIL})
        if resp.status_code == 200:
            log("Signup Init Success", True)
        else:
            log(f"Signup Init Failed: {resp.text}", False)
            return None
    except Exception as e:
        log(f"Signup Init Error: {e}", False)
        return None

    # 2. Signup Verify
    try:
        payload = {
            "email": TEST_EMAIL, 
            "otp": "123456", 
            "name": TEST_NAME, 
            "password": TEST_PASSWORD
        }
        resp = requests.post(f"{BASE_URL}/api/auth/signup-verify", json=payload)
        if resp.status_code == 200:
            data = resp.json()
            token = data.get("token")
            log("Signup Verify Success", True)
            return token
        else:
            log(f"Signup Verify Failed: {resp.text}", False)
            return None
    except Exception as e:
        log(f"Signup Verify Error: {e}", False)
        return None

def test_login_flow():
    log("Testing Login Flow...", True)
    try:
        payload = {"email": TEST_EMAIL, "password": TEST_PASSWORD}
        resp = requests.post(f"{BASE_URL}/api/auth/login", json=payload)
        if resp.status_code == 200:
            log("Login Success", True)
            return resp.json().get("token")
        else:
            log(f"Login Failed: {resp.text}", False)
            return None
    except Exception as e:
        log(f"Login Error: {e}", False)
        return None

def test_create_issue(token):
    log("Testing Issue Creation (Image Upload)...", True)
    if not os.path.exists(IMAGE_PATH):
        log(f"Test image not found at {IMAGE_PATH}", False)
        return None

    try:
        files = {
            'image': ('test_image.jpg', open(IMAGE_PATH, 'rb'), 'image/jpeg')
        }
        data = {
            'latitude': '28.6139',
            'longitude': '77.2090',
            'description': 'Automated Test Issue',
            'user_email': TEST_EMAIL
        }
        headers = {
             # 'Authorization': f'Bearer {token}' # API usually doesn't require auth for creation, but let's see
        }
        
        # Note: /api/issues is the endpoint in issues.py
        resp = requests.post(f"{BASE_URL}/api/issues", files=files, data=data, headers=headers)
        
        if resp.status_code == 200:
            result = resp.json()
            log("Issue Creation Success", True)
            log(f"Issue ID: {result.get('id')}", True)
            return result
        else:
            log(f"Issue Creation Failed: {resp.text}", False)
            return None
    except Exception as e:
        log(f"Issue Creation Error: {e}", False)
        return None

def main():
    print("--- STARTING END-TO-END TEST ---")
    
    print_debug_routes() # Check paths first

    # Auth Test
    token = test_signup_flow()
    if not token:
        # If signup fails (maybe user exists), try login
        token = test_login_flow()
    
    # Force verify issues even if token missing (expect 401 or 403, but NOT 404)
    if not token:
        log("Token missing, trying Issue Create anyway to check 404...", False)
        token = "DUMMY_TOKEN"

    # Issue Test
    issue_data = test_create_issue(token)
    
    if issue_data:
        log("✅ E2E TEST COMPLETED SUCCESSFULLY!", True)
    else:
        log("❌ E2E TEST FAILED", False)

if __name__ == "__main__":
    main()
