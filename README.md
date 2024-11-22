# Face Recognition Using LFW Dataset

## Problem Statement
Develop a CNN model to recognize faces using the [LFW dataset](https://www.kaggle.com/datasets/atulanandjha/lfwpeople). The project involves:
- Splitting the dataset into two categories: **Category A** and **Category B**.
- Training a CNN model to classify images into these categories.
- Selecting a random image and triggering an alert if it belongs to **Category A**.

## Project Overview
- This project uses the LFW (Labeled Faces in the Wild) dataset to train a Convolutional Neural Network (CNN) for binary face recognition.
- The classification system is scalable, using a **softmax function**, allowing for the addition of more classes in the future.
- **Category A** and **Category B** names:
  - **Category A**: George W. Bush, Tony Blair, Donald Rumsfeld, Ariel Sharon, Gerhard Schroeder, Jacques Chirac, Junichiro Koizumi, Luiz Inacio Lula da Silva, Hugo Chavez, Jean Chretien.
  - **Category B**: Colin Powell, Condoleezza Rice, Vladimir Putin, John Ashcroft, Pervez Musharraf, Alvaro Uribe, Abdullah Gul, Silvio Berlusconi, Paul Bremer, Kofi Annan.

## Features
- **CNN-Based Face Recognition**: Classifies faces into two categories.
- **Alert Mechanism**: Generates alerts if a random image belongs to **Category A**.
- **Flexible Architecture**: Can be extended to support multi-class classification using a softmax layer.

## Directory Structure
```plaintext
├── data/
│   └── lfw/              # (Place your LFW dataset here if training the model)
├── src/
│   ├── __init__.py       # (Marks src as a package)
│   ├── config.py         # (Configuration settings)
│   ├── preprocess.py     # (Preprocessing functions)
│   ├── model.py          # (Model architecture)
│   ├── train.py          # (Training logic)
│   ├── predict.py        # (Prediction logic)
├── main.py               # Main entry point
├── requirements.txt      # Dependencies
└── README.md             # Documentation
```

---

## Steps to Run the Project

### 1. Create and Activate Virtual Environment
Use `conda` to set up the Python environment:
```bash
conda create -n lfw python=3.9 -y
conda activate lfw
```

### 2. Install Dependencies
Install the required libraries:
```bash
pip install -r requirements.txt
```

---

## **Run the Application**

You have two options depending on your requirements:

### Option 1: Use Pretrained Model for Prediction
1. Ensure you have a pretrained model saved in the designated directory.
2. Start the prediction server:
   ```bash
   uvicorn app:app --reload
   ```
3. Test the application by sending requests to the prediction endpoint.

---

### Option 2: Train the Model
1. **Download the Dataset**: Download the LFW dataset for training:
   ```bash
   gdown --folder https://drive.google.com/drive/folders/18TbY20DdpXIguR6tSEKA7HoAHiC1jvfw
   ```
2. Run the training script:
   ```bash
   python train.py
   ```
3. After training, the model will be saved. You can then follow **Option 1** to use the newly trained model for predictions.

---

## Technologies Used
- **Deep Learning**: TensorFlow/Keras or PyTorch for CNN model development.
- **Python**: For scripting and automation.
- **FastAPI/Uvicorn**: For deploying the prediction API.
