# Face Recognition Using LFW Dataset

## Problem Statement
Train a cnn model to recognize face using this [data](https://www.kaggle.com/datasets/atulanandjha/lfwpeople).
split data in different different category e.g. category a and category b. then select random image and alert if images relate to category a.

## Project Overview
This project uses the LFW dataset to train a CNN model for face recognition. It classifies images into two categories (Category A and Category B) and alerts if a random image belongs to Category A.the are only 2 classes define category a cat b and i use softmax funtion so that we can increase number of classes for training 
    CATEGORY_A_NAMES = [
        "George_W_Bush", "Tony_Blair", "Donald_Rumsfeld", "Ariel_Sharon", 
        "Gerhard_Schroeder", "Jacques_Chirac", "Junichiro_Koizumi", 
        "Luiz_Inacio_Lula_da_Silva", "Hugo_Chavez", "Jean_Chretien"
    ]
    
    # 10 example names for Category B
    CATEGORY_B_NAMES = [
        "Colin_Powell", "Condoleezza_Rice", "Vladimir_Putin", "John_Ashcroft",
        "Pervez_Musharraf", "Alvaro_Uribe", "Abdullah_Gul", "Silvio_Berlusconi",
        "Paul_Bremer", "Kofi_Annan"
    ]
the classed contain 

## Directory Structure
```
├── data/
│   └── lfw/              # (Place your LFW dataset here)
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

## Steps to Run the Project

1. Create & Activate ENV:
```
conda create -n lfw python=3.9
conda activate lfw
```


2. Install dependencies:
```
pip install -r requirements.txt
```

5. Run (For Prediction from trained model):
```
uvicorn app:app --reload
```
3. Download dataset:
```
gdown --folder https://drive.google.com/drive/folders/18TbY20DdpXIguR6tSEKA7HoAHiC1jvfw
```
4. Run (For Training):
```
python train.py
```


## Features
- CNN model for binary face recognition.
- Alert mechanism for random image predictions.
