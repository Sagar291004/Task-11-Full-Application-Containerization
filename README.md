
# Task 11: Full Application Containerization

## Objective
To containerize the complete deep learning application,
including the trained model, Flask API and Streamlit UI.

## Technologies Used
- Python
- TensorFlow / Keras
- Flask
- Streamlit
- Docker
- Docker Compose

## Application Architecture
The Streamlit frontend accepts an image from the user.
It sends the image to the Flask API, which preprocesses
the image and uses the trained CNN model for prediction.
The prediction and confidence are returned to Streamlit.

## Project Components
- backend/app.py: Flask prediction API
- backend/task6_cnn_model.keras: Trained model
- backend/Dockerfile: Backend container configuration
- frontend/streamlit_app.py: Streamlit user interface
- frontend/Dockerfile: Frontend container configuration
- docker-compose.yml: Runs and connects both services

## Run the Application
Build the Docker images:

    docker compose build

Start the application:

    docker compose up -d

Check the containers:

    docker compose ps

Open the Streamlit UI:

    http://localhost:8501

## Testing Results
- Docker images built successfully.
- Flask API and Streamlit containers started.
- Image upload and prediction tested successfully.
- Flask API returned HTTP 200 for prediction requests.
- Example prediction: Automobile.
- Example confidence: 92.04%.

## Conclusion
The deep learning application was containerized using
Docker and Docker Compose. The Streamlit frontend
successfully communicated with the Flask API and
returned a model prediction.
