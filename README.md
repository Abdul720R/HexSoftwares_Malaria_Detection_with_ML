# HexSoftwares_Malaria_Detection_with_ML
# Malaria Cell Detection Using CNN & VGG19

## Project Overview

This project implements a deep learning model to classify malaria-infected and uninfected blood cell images. It includes two approaches:

1. **Transfer Learning** using a pre-trained **VGG19** model.
2. **Custom Convolutional Neural Network (CNN)** built from scratch.

The model is trained on labeled malaria cell images, and can predict whether a new image is “Infected” or “Uninfected.”

---

## Features

* Pre-trained **VGG19** with frozen layers for feature extraction
* Fully connected **Dense layers** for classification
* CNN from scratch with multiple convolutional and pooling layers
* Data preprocessing and augmentation using `ImageDataGenerator`
* Model evaluation with training/validation loss and accuracy plots
* Save and load trained models (`.h5` format)
* Predict single images with human-readable output

---

## Dataset

* **Training Set:** `Dataset/Train/` with subfolders for each class
* **Testing Set:** `Dataset/Test/` with subfolders for each class
* Images are resized to **224x224 pixels**

---

## Libraries / Tools Used

* Python 3.x
* TensorFlow & Keras
* NumPy
* Matplotlib
* Glob

---

3. **Predict a single image**

```python
python predict_image.py
```

* Load a trained model `.h5` file.
* Provide an image path to get the prediction: “Infected” or “Uninfected”.

---

## Results

* The model outputs **predicted probabilities** for each class.
* Predictions can be converted to **class labels** using `np.argmax()`.
* Example prediction:

```
Prediction: Uninfected
```

---

## Future Improvements

* Implement **other pre-trained models** like ResNet50 or EfficientNet.
* Use **larger datasets** for improved accuracy.
* Add **GUI or Web Application** for real-time malaria detection.

---
