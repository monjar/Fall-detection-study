"""
Example script to test the fall detection web app programmatically.
"""

import requests
import json
import time
from pathlib import Path

# Configuration
WEB_APP_URL = "http://127.0.0.1:5000"
VIDEO_PATH = Path("path/to/your/test_video.mp4")  # Change this

def test_web_app():
    """Test the fall detection web app."""
    
    print("=" * 80)
    print("Fall Detection Web App - API Test")
    print("=" * 80)
    
    # 1. Check health
    print("\n1. Checking server health...")
    try:
        response = requests.get(f"{WEB_APP_URL}/health")
        health = response.json()
        print(f"   Status: {health['status']}")
        print(f"   CUDA Available: {health['cuda_available']}")
        print(f"   Available Models: {health['available_models']}")
    except Exception as e:
        print(f"   Error: Could not connect to server. Is it running?")
        print(f"   Start the server with: cd web_app && python app.py")
        return
    
    # 2. Get available models
    print("\n2. Getting available models...")
    response = requests.get(f"{WEB_APP_URL}/api/models")
    models = response.json()
    print(f"   Found {len(models)} models:")
    for model in models:
        print(f"   - {model['name']}: {model['description']}")
    
    # 3. Upload video
    if not VIDEO_PATH.exists():
        print(f"\n3. Skipping upload - video not found: {VIDEO_PATH}")
        print("   Update VIDEO_PATH in this script to test upload and detection")
        return
    
    print(f"\n3. Uploading video: {VIDEO_PATH.name}...")
    with open(VIDEO_PATH, 'rb') as f:
        files = {'video': (VIDEO_PATH.name, f, 'video/mp4')}
        response = requests.post(f"{WEB_APP_URL}/api/upload", files=files)
        upload_result = response.json()
    
    if not upload_result.get('success'):
        print(f"   Error: {upload_result.get('error')}")
        return
    
    filename = upload_result['filename']
    print(f"   Uploaded successfully: {filename}")
    print(f"   Size: {upload_result['size']} bytes")
    
    # 4. Detect fall
    print("\n4. Running fall detection...")
    print("   This may take a few minutes...")
    
    for model in models[:1]:  # Test with first model
        model_id = model['id']
        print(f"\n   Testing with: {model['name']}")
        
        start_time = time.time()
        response = requests.post(
            f"{WEB_APP_URL}/api/detect",
            json={'filename': filename, 'model': model_id}
        )
        elapsed = time.time() - start_time
        
        result = response.json()
        
        if result.get('success'):
            print(f"   ✓ Detection completed in {elapsed:.1f}s")
            print(f"   Prediction: {result['label']}")
            print(f"   Confidence: {result['confidence']*100:.1f}%")
            
            if 'individual_predictions' in result:
                print(f"   Individual detector results:")
                for detector, data in result['individual_predictions'].items():
                    print(f"     - {detector}: {data['label']} ({data['confidence']*100:.1f}%)")
        else:
            print(f"   ✗ Error: {result.get('error')}")
    
    # 5. Cleanup
    print("\n5. Cleaning up...")
    response = requests.post(
        f"{WEB_APP_URL}/api/cleanup",
        json={'filename': filename}
    )
    print("   Cleanup completed")
    
    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)


if __name__ == '__main__':
    test_web_app()
