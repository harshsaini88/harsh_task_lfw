# preprocess module
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from src.config import Config

def load_data(dataset_path, img_size):
    images, labels = [], []
    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_path):
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
                img = cv2.resize(img, (img_size, img_size))  # Resize
                images.append(img)
                if person_name in Config.CATEGORY_A_NAMES:
                    labels.append(0)  # 0 for Category A
                else:
                    labels.append(1)  # 1 for Category B
    return np.array(images), np.array(labels)

def preprocess_data():
    # Load dataset
    X, y = load_data(Config.DATASET_PATH, Config.IMG_SIZE)

    # Normalize and reshape images
    X = X / 255.0  # Normalize
    X = X[..., np.newaxis]  # Add channel dimension (e.g., (n_samples, 64, 64, 1))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test
