import streamlit as st
import tensorflow as tf

st.set_option('depreciate.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation)

# fucntion to load the model
def load_model():
    
    model = tf.keras.models.load_model('C:/Users/Cash Crusaders/Desktop/My Portfolio/Projects/Data Science Projects/Deep Learning Project 1  - Corona Virus Detection ( Using Chest Xray Images)/model.h5')
    
    return model

model = load_model()
# write heading for the web app
st.title("Covid-19 Detection Web App")
st.write("### Classification Model")

file = st.file_uploader("Please upload a scanner image", type=["jpg", "png"])

import cv2
from PIL import Image, ImageOps
from keras.preprocessing import image
import numpy as np
import pandas as pd
from keras.applications.vgg16 import preprocess_input

# create a function to predict
def model_prediction(img, model):
    
    # convert the image into an array
    X = image.img_to_array(img)
    X = np.expand_dims(X, axis=0)
    
    # proprocess the image
    img_data = preprocess_input(X)
    outcome = ""
    classes  = model.predict(img_data)
    new_pred = np.argmax(classes, axis=1)
    if new_pred==[1]:
        outcome = "Prediction: Normal"
    else:
        outcome = "Prediction: Corona Virus"
        
    return outcome

if file is None:
    st.text("Please upload a scanner image")
else:
    img = Image.open(file)
    st.image(img, use_column_width = True)
    prediction = model_prediction(img, model)
    st.success(prediction)
    
    
if __name__ == '__main__':
    main()


