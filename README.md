# Face Recognition Using LFW Dataset

## Problem Statement
Train a cnn model to recognize face using this [data](https://www.kaggle.com/datasets/atulanandjha/lfwpeople).
split data in different different category e.g. Category A and Category B. then select random image and alert if images relate to Category A.

## Project Overview
This project uses the LFW dataset to train a CNN model for face recognition. It classifies images into two categories (Category A and Category B) and alerts if a random image belongs to Category A.

## Directory Structure
```
lfw_face_recognition/
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

4. Create Env:
```
conda create -n lfw python=3.9
```
3. Install dependencies:
```
pip install -r requirements.txt
```
3. Download dataset:
```
https://drive.google.com/drive/folders/18TbY20DdpXIguR6tSEKA7HoAHiC1jvfw?usp=sharing
```

4. Run (For Training):
```
python train.py
```
4. Run (For Prediction):
```
uvicorn app:app --reload
```

## Features
- CNN model for binary face recognition.
- Alert mechanism for random image predictions.
