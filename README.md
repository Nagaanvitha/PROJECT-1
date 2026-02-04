# PROJECT-1
Plant Disease Detection

# Plant Disease Classification using CNN
This project is a deep learning pipeline that classifies plant diseases from leaf images using a Convolutional Neural Network (CNN) built with Keras and TensorFlow. It uses the PlantVillage dataset to train a model capable of distinguishing between various plant diseases.

# Dataset
Source: Kaggle - PlantVillage Dataset

Classes: Multiple plant diseases across different species

Format: Images organized in subdirectories per disease category

# Requirements
Install the required packages via pip:

bash
Copy
Edit
pip install numpy opencv-python scikit-learn matplotlib keras tensorflow
# Project Structure
plaintext
Copy
Edit
.
├── cnn_model.h5                # Saved Keras model
├── label_transform.pkl         # Saved label binarizer
├── train_plant_disease.py      # Training script (your notebook's content as .py)
├── README.md                   # This file
└── /plantvillage               # Dataset directory
# Model Architecture
Input: 256x256 RGB images

3 Convolutional blocks with BatchNorm, MaxPooling, and Dropout

Fully connected layer with 1024 neurons

Softmax output for multi-class classification

# How to Use
1. Prepare Dataset
Download the dataset and place it in a folder named plantvillage/.

2. Train the Model
Run the training script:
bash
Copy
Edit
python train_plant_disease.py

3. Evaluate Accuracy
The script splits the dataset into train/test sets and outputs accuracy after training.

4. Model Output
Trained model saved as cnn_model.h5

Label encoder saved as label_transform.pkl

# Training & Validation Metrics
The script plots training vs validation accuracy and loss curves using matplotlib.

# Evaluation
The script uses:

Accuracy Score

Loss Curves

Optional: You can add classification report and confusion matrix

# Sample Prediction (Optional Extension)
To use the model for prediction:

python
Copy
Edit
from keras.models import load_model
import pickle
import cv2
import numpy as np

model = load_model('cnn_model.h5')
label_binarizer = pickle.load(open('label_transform.pkl', 'rb'))

def prepare_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256))
    img = img.astype("float") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

image = prepare_image("test_leaf.jpg")
prediction = model.predict(image)
label = label_binarizer.classes_[np.argmax(prediction)]
print(f"Predicted disease: {label}")
