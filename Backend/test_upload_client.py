from fastapi.testclient import TestClient
import os

from apps import app

client = TestClient(app)

file_path = os.path.join(os.path.dirname(__file__), 'test_upload.txt')
with open(file_path, 'rb') as f:
    files = {"file": ("test_upload.txt", f, "text/plain")}
    data = {"username": "testuser"}
    resp = client.post("/upload", files=files, data=data)

print('STATUS:', resp.status_code)
print('JSON:', resp.json())
