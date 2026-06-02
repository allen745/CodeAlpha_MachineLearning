# ✍️ Task 3 — Handwritten Character Recognition

A deep learning model using CNN to recognize handwritten digits from the MNIST dataset.

## 🎯 Objective
Identify handwritten digits (0-9) using Convolutional Neural Networks achieving ~99% accuracy.

## 🚀 Features
- MNIST dataset (60,000 training + 10,000 test images)
- Custom CNN architecture with 2 convolutional blocks
- Data augmentation (rotation, zoom, shift)
- Batch Normalization & Dropout for regularization
- Early Stopping & Learning Rate Reduction callbacks
- EDA with sample digit visualization
- Training history plots (accuracy & loss)
- Confusion matrix
- Per-class accuracy analysis
- Correct vs wrong prediction visualization
- Model saved for future inference

## 🧠 CNN Architecture
```
Input (28x28x1)
    ↓
Conv2D(32) → BatchNorm → Conv2D(32) → MaxPool → Dropout(0.25)
    ↓
Conv2D(64) → BatchNorm → Conv2D(64) → MaxPool → Dropout(0.25)
    ↓
Flatten → Dense(256) → BatchNorm → Dropout(0.5)
    ↓
Output: Softmax(10 classes)
```

## 📊 Results
- **Test Accuracy: ~99.2%**
- **Test Loss: ~0.025**

## 🛠️ Tech Stack
- Python 3
- TensorFlow / Keras
- NumPy
- Matplotlib & Seaborn
- Scikit-learn

## ▶️ Run
```bash
pip install -r requirements.txt
python handwritten_recognition.py
```

## 📁 Output Files
- `eda_digits.png` — Sample digits, class distribution, pixel analysis
- `cnn_results.png` — Training history, confusion matrix, per-class accuracy
- `handwritten_cnn_model.keras` — Saved trained model

## 👨‍💻 Author
Allen Stivanson Christian
CodeAlpha Machine Learning Internship — June 2026
Student ID: CA/DF1/110227
