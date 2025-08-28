# test_deriv_basic.py

import requests
import json
import time

def test_deriv_basic():
    """Basic test of Deriv API endpoints"""
    base_url = "https://api.deriv.com"
    
    print("🔍 **BASIC DERIV API TEST**")
    print("=" * 50)
    
    # Test 1: Basic connection
    print("1️⃣ Testing basic connection...")
    try:
        response = requests.get(base_url, timeout=10)
        print(f"   ✅ GET {base_url}: {response.status_code}")
        if response.status_code == 200:
            print(f"   📄 Response length: {len(response.text)} characters")
            print(f"   📄 Content type: {response.headers.get('content-type', 'N/A')}")
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
    
    # Test 2: Application endpoint
    print("\n2️⃣ Testing application endpoint...")
    try:
        response = requests.get(f"{base_url}/api/v1/application", timeout=10)
        print(f"   ✅ GET {base_url}/api/v1/application: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   📄 App ID: {data.get('application', {}).get('app_id', 'N/A')}")
        else:
            print(f"   📄 Response: {response.text[:200]}...")
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
    
    # Test 3: Trading times endpoint
    print("\n3️⃣ Testing trading times endpoint...")
    try:
        response = requests.get(f"{base_url}/api/v1/trading_times", timeout=10)
        print(f"   ✅ GET {base_url}/api/v1/trading_times: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            markets = data.get('trading_times', {}).get('markets', [])
            print(f"   📄 Markets count: {len(markets)}")
            if markets:
                print(f"   📄 First market: {markets[0].get('name', 'N/A')}")
        else:
            print(f"   📄 Response: {response.text[:200]}...")
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
    
    # Test 4: Ticks history endpoint (POST)
    print("\n4️⃣ Testing ticks history endpoint...")
    try:
        payload = {
            "ticks_history": "frxEURUSD",
            "adjust_start_time": 1,
            "count": 10,
            "end": int(time.time()),
            "start": int(time.time()) - 36000,  # 10 hours ago
            "style": "candles",
            "granularity": 3600
        }
        
        response = requests.post(
            f"{base_url}/api/v1/ticks_history",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        print(f"   ✅ POST {base_url}/api/v1/ticks_history: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            candles = data.get('candles', [])
            print(f"   📄 Candles count: {len(candles)}")
            if candles:
                print(f"   📄 First candle: O:{candles[0].get('open', 'N/A')} H:{candles[0].get('high', 'N/A')}")
        else:
            print(f"   📄 Response: {response.text[:200]}...")
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")

if __name__ == "__main__":
    test_deriv_basic()

