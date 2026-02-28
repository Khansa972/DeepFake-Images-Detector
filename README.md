# DeepFake Images Detector

## Overview
This project detects DeepFake images using a deep learning model. It has two parts:
1. **Training & Testing:** Trains the model and evaluates accuracy.
2. **GUI:** Allows users to upload images and check if they are Real or Fake.

## Dataset
Kaggle DeepFake and Real images dataset
**Preprocessing:** Resized to 128x128, normalized

## How to Use
1. Open "1_Train_Test_Validation.ipynb" to train and evaluate the model
2. Open "2_GUI.ipynb" to test images with the GUI
3. (Optional) Save/load trained model for faster GUI usage

## Requirements
- Python 3.x
- TensorFlow / Keras
- OpenCV
- PIL / Pillow
- ipywidgets
- matplotlib, seaborn, numpy, pandas

## Notes
- Predictions may be **inverted** automatically if the model is biased
- GUI supports real-time image upload and displays confidence scores
- **Link to Kaggle dataset:** https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images
