import streamlit as st
import tensorflow as tf

#st.set_option('depreciate.showfileUploaderEncoding', False)
# @st.cache(allow_output_mutation=True)

# fucntion to load the model
def load_model():
    
    model = tf.keras.models.load_model('model.h5')
    
    return model

model = load_model()
# write heading for the web app
st.title("Covid-19 Detection Web App")
st.write("### Classification Model")

file = st.file_uploader("Please upload a scanner image", type=["jpeg", "png"])

import cv2
from PIL import Image, ImageOps
from keras.preprocessing import image
import numpy as np
import pandas as pd
from keras.applications.vgg16 import preprocess_input

# create a function to predict
def model_prediction(img, model):
    
    size = (224,224)
    img = ImageOps.fit(img, size, Image.ANTIALIAS)
    X = image.img_to_array(img)
    img = cv2.cvtColor(X, cv2.COLOR_BGR2RGB)
    reshaped_img = img[np.newaxis,...]
    
    
    
   # X = image.img_to_array(img)
    
    #X = np.expand_dims(X, axis=0)
    #X = X[np.newaxis,np.newaxis]
    # preprocess the image
    #img_data = preprocess_input(X)
    
    #classes = ['Normal','Corona Virus']
    
    classes  = model.predict(reshaped_img)
    
    new_pred = np.argmax(classes, axis=1)
    if new_pred==[1]:
        outcome = "Prediction: Negative Covid-19"
    else:
        outcome = "Prediction: Positive Covid-19"
        
    return outcome

if file is None:
    st.text("Please upload a scanner image")
else:
    input_img = Image.open(file)
    st.image(input_img, use_column_width = True)
    prediction = model_prediction(input_img, model)
    st.success(prediction)
    
    



