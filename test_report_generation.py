"""
End-to-End Test for Report Generation Process
Tests the complete flow from image upload to report generation
"""

import requests
import json
import base64
from pathlib import Path

# Configuration
BASE_URL = "http://10.192.65.162:3001"
TEST_USER_EMAIL = "test@example.com"

def test_report_generation():
    """Test complete report generation flow"""
    
    print("=" * 80)
    print("🧪 TESTING END-TO-END REPORT GENERATION PROCESS")
    print("=" * 80)
    
    # Step 1: Create a test image (simple 1x1 pixel JPEG)
    print("\n📸 Step 1: Creating test image...")
    test_image_data = base64.b64decode(
        "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0a"
        "HBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIy"
        "MjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIA"
        "AhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEB"
        "AQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCwAB//2Q=="
    )
    
    # Save test image
    test_image_path = Path("test_issue.jpg")
    with open(test_image_path, "wb") as f:
        f.write(test_image_data)
    print(f"✅ Test image created: {test_image_path}")
    
    # Step 2: Prepare form data
    print("\n📋 Step 2: Preparing form data...")
    form_data = {
        'address': 'Test Street, Test City',
        'zip_code': '37062',
        'latitude': '36.1234',
        'longitude': '-86.5678',
        'user_email': TEST_USER_EMAIL
    }
    
    files = None
    try:
        files = {
            'image': ('test_issue.jpg', open(test_image_path, 'rb'), 'image/jpeg')
        }
    
        print(f"✅ Form data prepared:")
        for key, value in form_data.items():
            print(f"   - {key}: {value}")
    
        # Step 3: Send request to analyze endpoint
        print("\n🚀 Step 3: Sending request to /api/issues/analyze...")
        response = requests.post(
            f"{BASE_URL}/api/issues/analyze",
            data=form_data,
            files=files,
            timeout=120  # 2 minutes for AI processing
        )
        
        print(f"📡 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ SUCCESS! Report generated successfully!")
            
            # Parse response
            result = response.json()
            
            print("\n📊 RESPONSE DATA:")
            print(f"   - Success: {result.get('success')}")
            print(f"   - Issue ID: {result.get('issue_id')}")
            print(f"   - Confidence: {result.get('confidence')}%")
            print(f"   - Requires Admin Review: {result.get('requires_admin_review')}")
            print(f"   - Status: {result.get('status')}")
            print(f"   - Message: {result.get('message')}")
            
            # Check if report exists
            if 'report' in result:
                report = result['report']
                print("\n📝 REPORT DETAILS:")
                
                if 'issue_overview' in report:
                    overview = report['issue_overview']
                    print(f"   - Issue Type: {overview.get('type', 'N/A')}")
                    print(f"   - Severity: {overview.get('severity', 'N/A')}")
                    print(f"   - Category: {overview.get('category', 'N/A')}")
                    print(f"   - Summary: {overview.get('summary', 'N/A')[:100]}...")
                
                if 'template_fields' in report:
                    tf = report['template_fields']
                    print(f"\n📍 LOCATION INFO:")
                    print(f"   - Address: {tf.get('address', 'N/A')}")
                    print(f"   - ZIP Code: {tf.get('zip_code', 'N/A')}")
                    print(f"   - Timestamp: {tf.get('timestamp', 'N/A')}")
            
            print("\n" + "=" * 80)
            print("✅ END-TO-END TEST PASSED!")
            print("=" * 80)
            
            # Save full response for inspection
            with open("test_response.json", "w") as f:
                json.dump(result, f, indent=2)
            print(f"\n💾 Full response saved to: test_response.json")
            
        else:
            print(f"❌ FAILED! Status code: {response.status_code}")
            print(f"Response: {response.text}")
            
            print("\n" + "=" * 80)
            print("❌ END-TO-END TEST FAILED!")
            print("=" * 80)
            
    except requests.exceptions.Timeout:
        print("❌ TIMEOUT! Request took longer than 2 minutes")
        print("\n" + "=" * 80)
        print("❌ END-TO-END TEST FAILED!")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        print("\n" + "=" * 80)
        print("❌ END-TO-END TEST FAILED!")
        print("=" * 80)
    
    finally:
        # Close file if opened
        if files and 'image' in files:
            try:
                files['image'][1].close()
            except:
                pass
        
        # Cleanup
        if test_image_path.exists():
            test_image_path.unlink()
            print(f"\n🧹 Cleaned up test image")

if __name__ == "__main__":
    test_report_generation()
