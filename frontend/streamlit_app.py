
import os
import streamlit as st
import requests

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------

st.set_page_config(
    page_title="CIFAR-10 AI Classifier",
    page_icon="🤖",
    layout="wide"
)

# -------------------------------------------------
# Flask API Configuration
# -------------------------------------------------

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:5000/predict"
)

# -------------------------------------------------
# Title
# -------------------------------------------------

st.title("🤖 CIFAR-10 Image Classification")

st.write(
    "Upload an image and get a real-time prediction "
    "using the Flask API backend."
)

st.divider()

# -------------------------------------------------
# Image Upload
# -------------------------------------------------

st.subheader("📤 Upload an Image")

uploaded_file = st.file_uploader(
    "Choose a JPG, JPEG or PNG image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------------------------
# Prediction
# -------------------------------------------------

if uploaded_file is not None:

    st.success("Image uploaded successfully!")

    st.image(
        uploaded_file,
        caption="Uploaded Image",
        width=350
    )

    st.divider()

    if st.button("🔍 Predict", type="primary"):

        with st.spinner("Sending image to Flask API..."):

            try:
                files = {
                    "image": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )
                }

                # Send image to Flask API
                response = requests.post(
                    API_URL,
                    files=files,
                    timeout=120
                )

                # Check API response
                if response.status_code == 200:

                    result = response.json()

                    prediction = result["prediction"]
                    confidence = float(result["confidence"])

                    st.success(
                        f"Prediction: {prediction}"
                    )

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric(
                            "Predicted Class",
                            prediction
                        )

                    with col2:
                        st.metric(
                            "Confidence",
                            f"{confidence:.2f}%"
                        )

                else:
                    try:
                        result = response.json()
                        error_message = result.get(
                            "error",
                            "Prediction failed"
                        )
                    except ValueError:
                        error_message = response.text

                    st.error(
                        f"API Error ({response.status_code}): "
                        f"{error_message}"
                    )

            except requests.exceptions.Timeout:
                st.error(
                    "The Flask API took too long to respond. "
                    "Please try again."
                )

            except requests.exceptions.ConnectionError:
                st.error(
                    "Unable to connect to Flask API. "
                    "Please check whether the backend is running."
                )

            except Exception as e:
                st.error(
                    f"An error occurred: {str(e)}"
                )

else:
    st.info(
        "👆 Please upload an image to get a prediction."
    )

# -------------------------------------------------
# Footer
# -------------------------------------------------

st.divider()

st.caption(
    "Task 11 — Full Application Containerization"
)
