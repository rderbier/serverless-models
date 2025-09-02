#!/usr/bin/env python3
"""
Test script for the Sentence Transformer API
Run this script to test both local and deployed versions of the API
"""

import requests
import json
import time

def test_health_endpoint(base_url):
    """Test the health check endpoint"""
    print(f"Testing health endpoint: {base_url}/health")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_embedding_map_format(base_url):
    """Test embedding endpoint with map format"""
    print(f"\nTesting embedding endpoint (map format): {base_url}/embedding")
    
    payload = {
        "text1": "This is a sample sentence.",
        "text2": "Here is another example text.",
        "text3": "Machine learning is fascinating."
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{base_url}/embedding",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        end_time = time.time()
        
        print(f"Status: {response.status_code}")
        print(f"Response time: {end_time - start_time:.2f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Number of embeddings: {len(result)}")
            
            # Show first few dimensions of first embedding
            for key in result:
                embedding = json.loads(result[key])
                print(f"Embedding for '{key}': [{embedding[0]:.6f}, {embedding[1]:.6f}, ...] (length: {len(embedding)})")
                break
                
        else:
            print(f"Error response: {response.text}")
            
        return response.status_code == 200
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_embedding_kserve_format(base_url):
    """Test embedding endpoint with KServe format"""
    print(f"\nTesting embedding endpoint (KServe format): {base_url}/embedding")
    
    payload = {
        "instances": [
            "This is a sample sentence.",
            "Here is another example text.",
            "Machine learning is fascinating."
        ]
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{base_url}/embedding",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        end_time = time.time()
        
        print(f"Status: {response.status_code}")
        print(f"Response time: {end_time - start_time:.2f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            predictions = result["predictions"]
            print(f"Number of predictions: {len(predictions)}")
            
            # Show first few dimensions of first prediction
            if predictions:
                embedding = predictions[0]
                print(f"First embedding: [{embedding[0]:.6f}, {embedding[1]:.6f}, ...] (length: {len(embedding)})")
                
        else:
            print(f"Error response: {response.text}")
            
        return response.status_code == 200
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Run all tests"""
    print("Sentence Transformer API Test Suite")
    print("=" * 50)
    
    # Test local development server
    local_url = "http://localhost:5000"
    print(f"\n🧪 Testing LOCAL server: {local_url}")
    print("-" * 30)
    
    local_health = test_health_endpoint(local_url)
    local_map = test_embedding_map_format(local_url) if local_health else False
    local_kserve = test_embedding_kserve_format(local_url) if local_health else False
    
    print(f"\n📊 LOCAL Results:")
    print(f"  Health: {'✅ PASS' if local_health else '❌ FAIL'}")
    print(f"  Map Format: {'✅ PASS' if local_map else '❌ FAIL'}")
    print(f"  KServe Format: {'✅ PASS' if local_kserve else '❌ FAIL'}")
    
    # Uncomment and update URL to test deployed version
    # deployed_url = "https://your-app-name.onrender.com"
    # print(f"\n🚀 Testing DEPLOYED server: {deployed_url}")
    # print("-" * 40)
    # 
    # deployed_health = test_health_endpoint(deployed_url)
    # deployed_map = test_embedding_map_format(deployed_url) if deployed_health else False
    # deployed_kserve = test_embedding_kserve_format(deployed_url) if deployed_health else False
    # 
    # print(f"\n📊 DEPLOYED Results:")
    # print(f"  Health: {'✅ PASS' if deployed_health else '❌ FAIL'}")
    # print(f"  Map Format: {'✅ PASS' if deployed_map else '❌ FAIL'}")
    # print(f"  KServe Format: {'✅ PASS' if deployed_kserve else '❌ FAIL'}")

if __name__ == "__main__":
    main()
