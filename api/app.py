import matplotlib.pyplot as plt
import sys
# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
import pdb
from joblib import dump,load
import numpy as np
# import skimage
# from skimage.transform import resize
import pandas as pd
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import os

app = Flask(_name_)

# Define a function to load models
def load_model(model_type):
    model_directory = './models'
    model_file = f'Production_Model_{model_type}.joblib'
    model_path = os.path.join(model_directory, model_file)
    if os.path.exists(model_path):
        return load(model_path)
    else:
        return None

# Load the models
svm_model = load_model('svm')
lr_model = load_model('lr')
tree_model = load_model('tree')

@app.route('/predict/<model_type>', methods=['POST'])
def predict_digit(model_type):
    try:
        if model_type == 'svm' and svm_model is not None:
            model = svm_model
        elif model_type == 'lr' and lr_model is not None:
            model = lr_model
        elif model_type == 'tree' and tree_model is not None:
            model = tree_model
        else:
            return jsonify({'error': 'Invalid model type'})

        data = request.get_json()  # Parse JSON data from the request body
        image = data.get('image', [])

        digit = predict_digit_with_model(model, image)

        return jsonify({'result': digit})
    except Exception as e:
        return jsonify({'error': str(e)})

def predict_digit_with_model(model, image):
    try:
        # Convert the input list to a numpy array and preprocess for prediction
        img_array = np.array(image, dtype=np.float32).reshape(1, 28, 28, 1) / 255.0

        prediction = model.predict(img_array)
        digit = np.argmax(prediction)

        return digit
    except Exception as e:
        return str(e)

if _name_ == '_main_':
    app.run()