# SASL-sign-language-recognition-Project
Real-time sign language recognition system using TensorFlow/Keras with live webcam inference.
# Real-Time Sign Language Recognition (TensorFlow/Keras)

A real-time computer vision system for sign-language recognition using TensorFlow/Keras.
The system performs live webcam inference, applies robust image preprocessing, and converts
hand gestures into text with spell-checking and multilingual translation.

## Features
- 27-class classification (SPACE + A–Z)
- Transfer learning with MobileNetV2 / EfficientNet
- ROI-based preprocessing (Gaussian blur, adaptive threshold, Otsu)
- Real-time webcam inference
- GUI application for live interaction
- Evaluation using confusion matrices

## Tech Stack
- Python 3.10
- TensorFlow / Keras
- OpenCV
- NumPy, scikit-learn
- Tkinter / Streamlit
- pyspellchecker, requests

## Run
```bash
pip install -r requirements.txt
python app_sasl_gui.py

