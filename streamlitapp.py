#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageOps
import numpy as np
import os 


# In[2]:

model = load_model("milletnet2.h5") 


# In[3]:


import streamlit as st
st.write("""
         # Millet variety classifier
         """
        )
st.write("This is a simple image classification web app to classify millets: Finger millet, Foxtail millet, Proso millet")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])


# In[4]:


def import_and_predict(image_data,model):
    size = (256,256)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    my_image = image/255
    img_reshape = np.expand_dims(my_image, axis=0)
    prediction = model.predict(img_reshape)
    
    return prediction


# In[5]:


if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) == 0:
        st.write("It is Finger millet")
    elif np.argmax(prediction) == 1:
        st.write("It is Foxtail millet")
    else:
        st.write("It is Proso millet")
        
    st.text("Probaility (0: Finger millet, 1: Foxtail millet 2: Proso millet")
    st.write(prediction)


# In[ ]:




