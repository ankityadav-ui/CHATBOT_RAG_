from fastapi.testclient import TestClient
import traceback

from apps import app

client = TestClient(app)
payload = {"user_id": 1, "text": "hello"}
try:
    resp = client.post('/query', json=payload)
    print('STATUS', resp.status_code)
    try:
        print('JSON', resp.json())
    except Exception:
        print('TEXT', resp.text)
except Exception:
    traceback.print_exc()
