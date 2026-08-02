#!/usr/bin/env python3
import time
import requests
import json

def test_knowledge_search():
    print("Testing knowledge base search...")
    try:
        response = requests.post(
            "http://localhost:5003/api/search",
            json={"query": "如何优化数据库性能"}
        )
        if response.status_code == 200:
            result = response.json()
            print("✓ Knowledge search successful")
            print(f"  Found {len(result)} results")
            for i, item in enumerate(result[:3]):
                if isinstance(item, dict) and 'text' in item:
                    print(f"  {i+1}. {item['text'][:100]}...")
                else:
                    print(f"  {i+1}. {str(item)[:100]}...")
            return True
        else:
            print(f"✗ Knowledge search failed with status code: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Knowledge search error: {e}")
        return False

def test_login():
    print("\nTesting login functionality...")
    try:
        response = requests.post(
            "http://localhost:5003/api/auth/login",
            json={"username": "test", "password": "test123"}
        )
        if response.status_code == 200:
            result = response.json()
            print("✓ Login successful")
            print(f"  Response: {result}")
            return result.get('token') or result.get('access_token')
        else:
            print(f"✗ Login failed with status code: {response.status_code}")
            print(f"  Response: {response.text}")
            return None
    except Exception as e:
        print(f"✗ Login error: {e}")
        return None

def test_knowledge_search(token):
    print("Testing knowledge base search...")
    try:
        headers = {}
        if token:
            headers['Authorization'] = f'Bearer {token}'
        
        response = requests.post(
            "http://localhost:5003/api/search",
            json={"query": "如何优化数据库性能"},
            headers=headers
        )
        if response.status_code == 200:
            result = response.json()
            print("✓ Knowledge search successful")
            print(f"  Found {len(result)} results")
            for i, item in enumerate(result[:3]):
                if isinstance(item, dict) and 'text' in item:
                    print(f"  {i+1}. {item['text'][:100]}...")
                else:
                    print(f"  {i+1}. {str(item)[:100]}...")
            return True
        else:
            print(f"✗ Knowledge search failed with status code: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Knowledge search error: {e}")
        return False

def main():
    print("Starting system tests...\n")
    
    # Wait for the server to start
    print("Waiting for server to start...")
    time.sleep(5)
    
    # Test login first
    token = test_login()
    login_success = token is not None
    
    # Test knowledge search with token
    search_success = test_knowledge_search(token)
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"Login: {'✓ PASSED' if login_success else '✗ FAILED'}")
    print(f"Knowledge search: {'✓ PASSED' if search_success else '✗ FAILED'}")
    
    if search_success and login_success:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    exit(main())