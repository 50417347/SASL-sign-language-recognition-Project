## SOUTH AFRICAN  SIGN LANGUAGE  INTO  OTHER AFRICAN LANGUAGES 
## Problem statement 
Many people in South Africa are unable to understand sign language, creating significant communication barriers for deaf and hard-of-hearing individuals. There is a need for an automated system that can recognize sign language gestures and translate them into text and spoken African languages in real time. This project addresses that need by capturing sign language gestures via a camera, converting them into text, forming words and sentences, and enabling translation into multiple African languages based on user selection.

# Real-Time Sign Language Recognition (TensorFlow / Keras)
A real-time computer vision system for sign-language recognition built using
TensorFlow/Keras. The system performs live webcam inference, applies a robust
image-quality preprocessing pipeline, and converts hand gestures into text with
spell-checking and multilingual translation support.

This project demonstrates an end-to-end production-style computer vision
workflow: data collection, preprocessing, model training, evaluation, and
real-time deployment.

---

## Key Features
- Multi-class image classification (**27 classes: SPACE + A–Z**)
- Transfer learning with **MobileNetV2 / EfficientNet**
- Robust preprocessing pipeline:
  - ROI extraction
  - Gaussian blur
  - Adaptive thresholding
  - Otsu binarization
- Real-time webcam inference with temporal stability logic
- GUI application for live interaction
- Spell-check suggestions during word formation
- Multilingual translation via API integration
- Model evaluation with confusion matrix and per-class metrics

---

##  Model Architecture
- Input size: **224 × 224**
- Input type: Grayscale images replicated to 3 channels
- Backbone: **MobileNetV2** (ImageNet pretrained)
- Training strategy:
  - Frozen backbone for baseline training
  - Fine-tuning with low learning rate
  - EarlyStopping and learning-rate scheduling

---

##  Evaluation & Results
- Validation accuracy: **~99%**
- Classes: **27**
- Real-time inference: **webcam-based**
- Evaluation artifacts:
  - Confusion matrix
  - Normalized confusion matrix
  - Per-class precision, recall, and F1-score

All evaluation outputs are available in the `/reports` directory.

---

## 📂 Dataset
The dataset consists of grayscale hand-gesture images organized into
27 classes (SPACE + A–Z). Images were collected using a custom webcam pipeline
with consistent ROI placement and image preprocessing (blur + adaptive threshold).

Due to size and privacy considerations, the dataset is **not included** in this
repository. The training and evaluation pipelines are fully reproducible if
the dataset is provided.

---

## 🛠️ Tech Stack
- Python 3.10
- TensorFlow / Keras
- OpenCV
- NumPy, scikit-learn
- Tkinter / Streamlit
- pyspellchecker
- requests
## Project Structure
sasl-sign-language-recognition/
├── app_sasl_gui.py
├── requirements.txt
├── models/
│   └──sasl_mobilenet_v2.keras
    ├──EfficientNet_SASL_letters_best.keras
├── notebooks/
│   ├── 01_train.ipynb
    ├── 02_Processing.ipynb
│   └── 03_evaluate.ipynb
├── reports/
│   ├── confusion_matrix.png
│   └── confusion_matrix_normalized.png
    └── Evaluation_Metrics.png
├── screenshots/
│   └── gui_live.png
## Production Notes
-Model exported in Keras format for deployment
-Designed with consistent preprocessing between training and inference
-Suitable for extension to TF Lite or TensorFlow Serving
-Modular code structure for maintainability
## Application Features (GUI)
Live Interface
Webcam feed with ROI overlay
Real-time prediction & confidence score
Processed ROI preview (thresholded image)
Text Construction
Automatic letter detection
Word building
Sentence construction
Space and backspace controls
Full-stop detection to finalize sentences
Spell Checking
Dynamic word suggestions
User-selected correction options
##Translation
Translates completed sentences into African languages
API-ready design (translation layer is modular)
Clear input/output textboxes

## ▶️ Running the Application
```bash
pip install -r requirements.txt
python app_sasl_gui.py
## Error analysis
I performed error analysis using confusion matrices and real-time testing. Most misclassifications occurred between visually similar signs or when the ROI did not contain a clear hand. I addressed this by enforcing consistent preprocessing, adding a background/SPACE class, applying confidence thresholds, and using temporal stability logic to reduce false positives.

**## Author
Developed as part of an MSc Computer Science project and extended as a
production-ready computer vision portfolio project.

