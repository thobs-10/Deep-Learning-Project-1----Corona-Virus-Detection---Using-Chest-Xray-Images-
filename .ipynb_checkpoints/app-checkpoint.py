# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 19:08:26 2022

@author: Thobela Sixpence
"""

# import libraries
import sys
import os
import glob
import re
import numpy as np

# Keras Libraries
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# tensorflow libraries
import tensorflow as tf


# Flask Libraries
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# load the model
model = tf.keras.models.load_model(r'C:\Users\Cash Crusaders\Desktop\My Portfolio\Projects\Data Science Projects\Deep Learning Project 1  - Corona Virus Detection ( Using Chest Xray Images)\model.h5')

# create a function to predict the outcome
def model_prediction(img_path, model):
    
    # get the image and resize it
    img = image.load_img(img_path, target_size=(150, 150, 3))
    
    # convert the image into an array
    X = image.img_to_array(img)
    X = np.expand_dims(X, axis=0)
    
    # proprocess the image
    img_data = preprocess_input(X)
    
    
    classes = model.predict(img_data)
    new_pred = np.argmax(classes, axis=1)
    if new_pred==[1]:
    
        outcome = "Prediction: Normal"
    else:
        outcome = "Prediction: Covid-19"
        
        
    return outcome

    
@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        prediction = model_prediction(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        #result = str(pred_class[0][0][1])               # Convert to string
        return prediction
    return None

if __name__ == '__main__':
    app.run(debug=True)
