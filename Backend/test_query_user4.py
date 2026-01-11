from fastapi.testclient import TestClient
from apps import app

client = TestClient(app)
resp = client.post('/query', json={"user_id": 4, "text": "test query for user 4"})
print('STATUS', resp.status_code)
try:
    print('JSON', resp.json())
except Exception:
    print('TEXT', resp.text)
