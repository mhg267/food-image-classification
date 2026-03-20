# Food Image Classification (CNN from Scratch)

This project builds a Convolutional Neural Network (CNN) from scratch to classify food images.

The main goal is to understand how CNN works without using transfer learning.

---

## 📊 Result

* Train Accuracy: ~0.93
* Validation Accuracy: ~0.83

---

## 📂 Dataset

* Source: Kaggle Food Image Dataset
* Number of classes: 34
* Link: https://www.kaggle.com/datasets/harishkumardatalab/food-image-classification-dataset

Data is split into:

* Train: 80%
* Validation: 10%
* Test: 10%

---

## 🧠 Model

The model includes:

* Multiple Conv2D + BatchNorm + LeakyReLU blocks
* MaxPooling layers
* Adaptive Average Pooling
* Fully connected layers with Dropout

Model file: `model_architecture.py`

---

## ⚙️ How to run

### Install dependencies

```bash
pip install torch torchvision scikit-learn opencv-python tqdm
```

### Train model

```bash
python train_model.py --root /path/to/dataset
```

### Inference

Evaluate model:

```bash
python mode_inference.py --best_model_path trained_model/best_model.pth --root /path/to/dataset
```

Predict single image:

```bash
python mode_inference.py --image_path path/to/image.jpg
```

---

## 🔧 Features

* Train from scratch (no pretrained model)
* Data augmentation
* Early stopping
* Learning rate scheduler
* Save best model

---
## 📦 Trained Model

The trained model is not included in this repository due to file size limitations.

You can:
- Train the model using the provided script
- Or contact me if you need the trained weights

---
## 📌 Notes

* The model shows some overfitting (train acc higher than val acc)
* Can be improved by using transfer learning or tuning hyperparameters

---

## 👤 Author

Minh Hung Le
