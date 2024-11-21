from src.train import train_model
from src.predict import alert_random_image
from src.config import Config
from tensorflow.keras.models import load_model
import os
from PIL import Image
import numpy as np
import cv2

# Define paths
MODEL_PATH = "face_recognition_model.h5"

def preprocess_uploaded_image(image_path, img_size):
    # Load and preprocess the image
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = np.array(image)
    image = cv2.resize(image, (img_size, img_size))  # Resize
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def main():
    print("Face Recognition Using LFW Dataset")
    print("Choose an action:")
    print("1. Predict")
    print("2. Train")

    # Get user choice
    choice = input("Enter your choice (1/2): ")

    if choice == "1":
        print("You selected: Predict")
        
        if os.path.exists(MODEL_PATH):
            print("Loading the model...")
            model = load_model(MODEL_PATH)
            print("Model loaded successfully!")

            # Get the image path from user
            image_path = input("Enter the path of the image for prediction: ")
            
            if os.path.exists(image_path):
                # Preprocess the image
                processed_image = preprocess_uploaded_image(image_path, Config.IMG_SIZE)
                
                # Predict
                print("Running prediction...")
                prediction = model.predict(processed_image)
                predicted_label = np.argmax(prediction)

                # Determine the category or none
                if predicted_label == 0:
                    print("image belong to category A.")
                elif predicted_label == 1:
                    print("image didn't belong to category A.")
                else:
                    print("image didn't belong to any category.")
                
                print("Prediction complete!")
            else:
                print("Error: Image file not found.")
        else:
            print("No pre-trained model found! Please train the model first.")

    elif choice == "2":
        print("You selected: Train")
        
        # Get the number of epochs from user
        epochs = int(input("Enter number of epochs for training: "))
        
        # Train the model
        print("Training the model...")
        train_model(epochs=epochs)
        print("Model trained successfully! The model has been saved.")

    else:
        print("Invalid choice. Please select 1 or 2.")

if __name__ == "__main__":
    main()
