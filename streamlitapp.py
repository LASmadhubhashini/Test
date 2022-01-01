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
st.write("Millet is a common term that includes cultivated grasses with very small seeds and is cultivated at several of locations in different agro-ecological regions in Sri Lanka. The proso, foxtail, and finger millets are common among the other millet types growing in Sri Lanka.")
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
        st.write("It is Finger millet.")
        st.write("In Sri lanka commonly known as ‘’kurakkan’’ or ‘’Kurahan”.Finger millet consists higher amount of calcium,dietary fiber,polyphenols, gluten-free and low protein content, crude fat content.")
        st.write("Finger millets help to control diabetes,strengthening bones,build strong immunity,good digestion, and they are good for celiac patients.")
    elif np.argmax(prediction) == 1:
        st.write("It is Foxtail millet.")
        st.write("In Sri Lanka commonly known as “Thana hal”.Foxtail millet consists starch, polyphenols, dietary fiber, vitamins and glueten-free.")
        st.write("Foxtail milets help to control diabetes,maintain strong bones,build strong immunity,good digestion and they are good for celiac patients.")
    else:
        st.write("It is Proso millet.")
        st.write("In Sri Lanka commonly known as ‘’Meneri’’.Proso millet consists high lecithin, protein,starch, polyphenols, dietary fiber, fatty acids, vitamins like B-complex vitamins, minerals and glueten-free.")
        st.write("Proso milets help to control diabetes,support neural health system,good digestion and they are good for celiac patients.")
    st.text("Probaility (0: Finger millet, 1: Foxtail millet 2: Proso millet)")
    st.write(prediction)


# In[ ]:




