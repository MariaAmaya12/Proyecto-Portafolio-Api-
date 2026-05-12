from fastapi.testclient import TestClient

from backend.main import app


def test_db_health_returns_ok():
    client = TestClient(app)

    response = client.get("/db/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["database"] == "sqlite"
