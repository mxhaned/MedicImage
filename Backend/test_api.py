import requests
import base64
import json
import os
from datetime import datetime

# API base URL
API_BASE_URL = "http://localhost:5000/api"

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_model_info():
    """Test the model info endpoint"""
    print("\nTesting model info...")
    try:
        response = requests.get(f"{API_BASE_URL}/model-info")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model info: {data}")
            return True
        else:
            print(f"❌ Model info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Model info error: {e}")
        return False

def test_classification_with_dummy_image():
    """Test classification with a dummy image"""
    print("\nTesting classification with dummy image...")
    
    # Create a simple test image (1x1 pixel, red)
    from PIL import Image
    import io
    
    # Create a simple test image
    test_image = Image.new('RGB', (100, 100), color='red')
    
    # Convert to base64
    buffer = io.BytesIO()
    test_image.save(buffer, format='PNG')
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # Add data URL prefix
    image_data_url = f"data:image/png;base64,{image_base64}"
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/classify",
            json={"image": image_data_url},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Classification successful!")
            print(f"Primary condition: {data['primary_condition']}")
            print(f"Confidence: {data['confidence']:.2%}")
            print("\nAll predictions:")
            for condition, probability in data['predictions'].items():
                print(f"  {condition}: {probability:.2%}")
            return data  # Return the data for report testing
        else:
            print(f"❌ Classification failed: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Classification error: {e}")
        return None

def test_report_generation(classification_result, image_data_url):
    """Test report generation with classification results"""
    print("\nTesting report generation...")
    
    if not classification_result:
        print("❌ No classification result available for report testing")
        return False
    
    try:
        # Prepare analysis data for report
        analysis_data = {
            "predictions": classification_result['predictions'],
            "primary_condition": classification_result['primary_condition'],
            "confidence": classification_result['confidence'],
            "recommendations": [
                "Use a gentle cleanser twice daily",
                "Apply appropriate treatment as recommended",
                "Consult with a dermatologist for severe cases"
            ],
            "products": [
                {
                    "name": "Gentle Cleanser",
                    "brand": "CeraVe",
                    "rating": 4.7,
                    "price": "$14.99"
                },
                {
                    "name": "Moisturizer",
                    "brand": "Neutrogena",
                    "rating": 4.6,
                    "price": "$16.99"
                }
            ]
        }
        
        # Test report generation
        response = requests.post(
            f"{API_BASE_URL}/generate-report",
            json={
                "image": image_data_url,
                "analysis_data": analysis_data,
                "patient_name": "Mr Ramzi Houidi"
            },
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            # Check if response is PDF
            content_type = response.headers.get('content-type', '')
            if 'application/pdf' in content_type:
                print("✅ Report generation successful!")
                print(f"PDF size: {len(response.content)} bytes")
                print(f"Content-Type: {content_type}")
                
                # Save the PDF for inspection
                filename = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                with open(filename, 'wb') as f:
                    f.write(response.content)
                print(f"✅ Report saved as: {filename}")
                return True
            else:
                print(f"❌ Unexpected content type: {content_type}")
                return False
        else:
            print(f"❌ Report generation failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Report generation error: {e}")
        return False

def test_classification_with_real_image(image_path):
    """Test classification with a real image file"""
    print(f"\nTesting classification with real image: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return False
    
    try:
        # Read and encode the image
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            # Determine image format
            if image_path.lower().endswith('.png'):
                mime_type = 'image/png'
            elif image_path.lower().endswith('.jpg') or image_path.lower().endswith('.jpeg'):
                mime_type = 'image/jpeg'
            else:
                mime_type = 'image/png'  # Default
            
            image_data_url = f"data:{mime_type};base64,{image_base64}"
        
        response = requests.post(
            f"{API_BASE_URL}/classify",
            json={"image": image_data_url},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Classification successful!")
            print(f"Primary condition: {data['primary_condition']}")
            print(f"Confidence: {data['confidence']:.2%}")
            print("\nAll predictions:")
            for condition, probability in data['predictions'].items():
                print(f"  {condition}: {probability:.2%}")
            return data
        else:
            print(f"❌ Classification failed: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Classification error: {e}")
        return None

def main():
    """Run all tests"""
    print("🧪 Testing MedicImage API")
    print("=" * 50)
    
    # Test health check
    health_ok = test_health_check()
    
    # Test model info
    model_ok = test_model_info()
    
    # Test classification with dummy image
    classification_result = test_classification_with_dummy_image()
    dummy_ok = classification_result is not None
    
    # Test report generation
    report_ok = False
    if classification_result:
        # Create dummy image for report testing
        from PIL import Image
        import io
        test_image = Image.new('RGB', (100, 100), color='red')
        buffer = io.BytesIO()
        test_image.save(buffer, format='PNG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        image_data_url = f"data:image/png;base64,{image_base64}"
        
        report_ok = test_report_generation(classification_result, image_data_url)
    
    # Test with real image if available
    test_images = [
        "test_image.jpg",
        "test_image.png",
        "sample.jpg",
        "sample.png"
    ]
    
    real_image_ok = False
    for image_path in test_images:
        if os.path.exists(image_path):
            real_image_result = test_classification_with_real_image(image_path)
            real_image_ok = real_image_result is not None
            break
    
    if not any(os.path.exists(img) for img in test_images):
        print("\nℹ️  No test images found. Place a test image in the Backend directory to test with real images.")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print(f"Health Check: {'✅ PASS' if health_ok else '❌ FAIL'}")
    print(f"Model Info: {'✅ PASS' if model_ok else '❌ FAIL'}")
    print(f"Dummy Image Classification: {'✅ PASS' if dummy_ok else '❌ FAIL'}")
    print(f"Report Generation: {'✅ PASS' if report_ok else '❌ FAIL'}")
    if any(os.path.exists(img) for img in test_images):
        print(f"Real Image Classification: {'✅ PASS' if real_image_ok else '❌ FAIL'}")
    
    all_tests_passed = health_ok and model_ok and dummy_ok and report_ok
    if real_image_ok is not None:
        all_tests_passed = all_tests_passed and real_image_ok
    
    if all_tests_passed:
        print("\n🎉 All tests passed! The API is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please check the API server and try again.")

if __name__ == "__main__":
    main() 