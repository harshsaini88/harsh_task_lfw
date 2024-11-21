from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import cv2
import shutil
from src.config import Config

# Define paths
MODEL_PATH = "face_recognition_model.h5"

# Initialize FastAPI app
app = FastAPI()

# Mount static files for serving the frontend
app.mount("/static", StaticFiles(directory="static"), name="static")


def preprocess_uploaded_image(image_path, img_size):
    """
    Load and preprocess the uploaded image for prediction.
    - Converts to grayscale
    - Resizes to the required dimensions
    - Normalizes pixel values
    - Adds batch and channel dimensions
    """
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = np.array(image)
    image = cv2.resize(image, (img_size, img_size))  # Resize
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """
    Serve the HTML frontend for uploading an image and getting predictions.
    """
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Predict the category of an uploaded image using the pre-trained model.
    """
    try:
        # Save the uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Check if the model exists
        if not os.path.exists(MODEL_PATH):
            return JSONResponse(content={"error": "No pre-trained model found!"}, status_code=400)

        # Load the model
        model = load_model(MODEL_PATH)

        # Preprocess the uploaded image
        processed_image = preprocess_uploaded_image(temp_path, Config.IMG_SIZE)

        # Predict
        prediction = model.predict(processed_image)
        predicted_label = np.argmax(prediction)

        # Cleanup temp file
        os.remove(temp_path)

        # Determine category
        if predicted_label == 0:
            category = "image belong to category A."
        elif predicted_label == 1:
            category = "image didn't belong to  category A."
        else:
            category = "Unknown"

        return JSONResponse(content={"prediction": category})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
