import pytest
from unittest.mock import patch
from api.app import app, predict_digit

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

@patch('api.app.predict_digit')
def test_post_predict(mock_predict_digit, client):
    # Iterate over digits 0 to 9
    for digit in range(10):
        # Set up the mock to return the current digit for testing
        mock_predict_digit.return_value = digit

        # Simulate a request to the /predict endpoint
        response = client.post("/predict", json={"image1": [0.1] * 784, "image2": [0.1] * 784})

        # Assert the response status code
        assert response.status_code == 200

        # Assert the predicted digit
        assert response.json['result'] == (digit == digit)
