# DeepFake Images Detector

## Overview
This project detects DeepFake images using a deep learning model. It has two parts:

1. **Training & Testing:** Trains the model and evaluates accuracy.  
2. **GUI:** Allows users to upload images and check if they are Real or Fake.

## Dataset
- Kaggle DeepFake and Real images dataset  
- **Preprocessing:** images resized to 128x128 and normalized  
- **Kaggle link:** [DeepFake & Real Images Dataset](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images)

## How to Use
1. Open the Python scripts in the `notebooks/` folder:
   - "1_Train_Test_Validation.py" – for training and evaluation  
   - "2_GUI.py" – for testing images with the GUI  
2. Make sure your dataset is organized correctly:
Dataset/
Train/
Fake/
Real/
Validation/
Fake/
Real/
Test/
Fake/
Real/

3. Run the scripts in order:
- First, train the model or load a pre-trained model.  
- Then use the GUI script to test images.  
4. Optionally, save/load the trained model for faster usage.

## Requirements
- Python 3.x  
- TensorFlow / Keras  
- OpenCV  
- PIL / Pillow  
- ipywidgets  
- matplotlib, seaborn, numpy, pandas

## Notes
- Predictions may be **inverted automatically** if the model is biased.  
- GUI supports real-time image upload and displays confidence scores.
