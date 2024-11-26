import pytest
from io import BytesIO
from src.app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def create_test_file(content, filename):
    return (BytesIO(content), filename)

def test_classify_file_no_file(client):
    response = client.post('/classify_file')
    assert response.status_code == 400
    assert b"No file part in the request" in response.data

def test_classify_file_empty_filename(client):
    test_file = create_test_file(b"test content", "")
    response = client.post('/classify_file', 
                         data={'file': test_file})
    assert response.status_code == 400
    assert b"No selected file" in response.data

def test_classify_file_success(client):
    test_file = create_test_file(b"test content", "test.txt")
    response = client.post('/classify_file',
                         data={'file': (BytesIO(b"test content"), "test.txt")})
    assert response.status_code == 200
    assert "file_class" in response.get_json()

def test_classify_file_server_error(client, monkeypatch):
    # Mock classifier to raise an exception
    def mock_classify(*args):
        raise Exception("Test error")
    
    monkeypatch.setattr("src.app.classifier.classify", mock_classify)
    
    test_file = create_test_file(b"test content", "test.txt")
    response = client.post('/classify_file',
                         data={'file': (BytesIO(b"test content"), "test.txt")})
    assert response.status_code == 500
    assert b"Internal server error" in response.data