"""
API Testing Examples
Use these examples to test the Zyntra API endpoints.
"""

import requests
import json

# Configuration
BASE_URL = "http://localhost:8000"
TOKEN = None  # Will be set after login


def print_response(response):
    """Pretty print API response."""
    print(f"Status: {response.status_code}")
    try:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except:
        print(f"Response: {response.text}")
    print("-" * 80)


def test_health_check():
    """Test health check endpoint."""
    print("\n1. Testing Health Check...")
    response = requests.get(f"{BASE_URL}/health")
    print_response(response)


def test_register_admin():
    """Test admin registration."""
    print("\n2. Testing Admin Registration...")
    data = {
        "email": "testadmin@zyntra.com",
        "username": "testadmin",
        "password": "password123",
        "full_name": "Test Admin",
        "is_super_admin": False
    }
    response = requests.post(f"{BASE_URL}/api/auth/admin/register", json=data)
    print_response(response)


def test_login_admin():
    """Test admin login."""
    global TOKEN
    print("\n3. Testing Admin Login...")
    data = {
        "email": "testadmin@zyntra.com",
        "password": "password123"
    }
    response = requests.post(f"{BASE_URL}/api/auth/login", json=data)
    print_response(response)
    
    if response.status_code == 200:
        TOKEN = response.json().get("access_token")
        print(f"✅ Token saved: {TOKEN[:20]}...")


def test_get_current_admin():
    """Test getting current admin info."""
    print("\n4. Testing Get Current Admin...")
    if not TOKEN:
        print("❌ No token available. Please login first.")
        return
    
    headers = {"Authorization": f"Bearer {TOKEN}"}
    response = requests.get(f"{BASE_URL}/api/admins/me", headers=headers)
    print_response(response)


def test_create_user():
    """Test creating a user under admin."""
    print("\n5. Testing Create User/Employee...")
    if not TOKEN:
        print("❌ No token available. Please login first.")
        return
    
    data = {
        "email": "employee1@zyntra.com",
        "username": "employee1",
        "password": "password123",
        "full_name": "Employee One"
    }
    headers = {"Authorization": f"Bearer {TOKEN}"}
    response = requests.post(f"{BASE_URL}/api/users/", json=data, headers=headers)
    print_response(response)


def test_list_users():
    """Test listing all users under admin."""
    print("\n6. Testing List Users/Employees...")
    if not TOKEN:
        print("❌ No token available. Please login first.")
        return
    
    headers = {"Authorization": f"Bearer {TOKEN}"}
    response = requests.get(f"{BASE_URL}/api/users/", headers=headers)
    print_response(response)


def test_get_admin_with_employees():
    """Test getting admin with employees."""
    print("\n7. Testing Get Admin With Employees...")
    if not TOKEN:
        print("❌ No token available. Please login first.")
        return
    
    headers = {"Authorization": f"Bearer {TOKEN}"}
    response = requests.get(f"{BASE_URL}/api/admins/me/employees", headers=headers)
    print_response(response)


def run_all_tests():
    """Run all API tests."""
    print("=" * 80)
    print("ZYNTRA API TESTING")
    print("=" * 80)
    
    test_health_check()
    test_register_admin()
    test_login_admin()
    test_get_current_admin()
    test_create_user()
    test_list_users()
    test_get_admin_with_employees()
    
    print("\n" + "=" * 80)
    print("✅ All tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        run_all_tests()
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Cannot connect to API server.")
        print("Please make sure the server is running at http://localhost:8000")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
