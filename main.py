import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, Dropout, MaxPool2D, Dense, Flatten
from tensorflow.keras.models import Model

DIM = (256, 256)

def image_resize(image, DIM=(256,256)):
    # resize the image
    image = cv2.resize(image, DIM, interpolation = cv2.INTER_AREA)
    return image


def preprocess_image(image):
    resized_image = image_resize(np.array(image), DIM=DIM)
    expanded_dim_image = tf.expand_dims(resized_image, 0)
    return expanded_dim_image


def load_model():

    inputs = Input(shape=(DIM[0], DIM[1], 3))
    conv1 = Conv2D(8, (3,3), activation='relu')(inputs)
    pool = MaxPool2D()(conv1)
    conv2 = Conv2D(16, (3,3), activation='relu')(pool)
    pool = MaxPool2D()(conv2)
    conv3 = Conv2D(16, (3,3), activation='relu')(pool)
    pool = MaxPool2D()(conv3)
    conv4 = Conv2D(16, (3,3), activation='relu')(pool)
    flatten = Flatten()(conv4)
    layer = Dense(256, activation='relu')(flatten)
    dropout = Dropout(0.1)(layer)
    layer = Dense(128, activation='relu')(dropout)
    dropout = Dropout(0.1)(layer)
    layer = Dense(64, activation='relu')(dropout)

    output = Dense(10, activation='softmax')(layer)

    model = Model(inputs=inputs, outputs=output)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
    model.load_weights('./digit_recognizer_model.h5')
    return model


def predict_digit(preprocessed_image, model):
    result = model.predict(preprocessed_image)
    digit = np.argmax(result)
    return digit

model = load_model()



st.markdown(
    """
    <style>
    .reportview-container {
        background: url("https://www.publicdomainpictures.net/pictures/260000/velka/degrade-texture-1-15294231389TA.jpg")
    }
    </style>
    """, unsafe_allow_html=True)


    
# Header of streamlit webpage
header_title = '<p style="color:Black; font-size: 40px;">Hand written digit recognition</p>'

st.markdown(header_title, unsafe_allow_html=True)

header_text_1 = '<p style="color:Black; font-size: 20px;">1.  This is AI based web-app.</p>'
header_text_2 = '<p style="color:Black; font-size: 20px;">2.  It tells which numerical digit is in the picture.</p>'
header_text_3 = '<p style="color:Black; font-size: 20px;">3.  The image dimension ratio should be 1:1 for best result.</p>'
header_text_4 = '<p style="color:Black; font-size: 20px;">4.  The digit written by hand should be bold and large.</p>'
header_text_5 = '<p style="color:Black; font-size: 20px;">5.  Use thick marker to write digit for better result.</p>'

st.markdown(header_text_1, unsafe_allow_html=True)
st.markdown(header_text_2, unsafe_allow_html=True)
st.markdown(header_text_3, unsafe_allow_html=True)
st.markdown(header_text_4, unsafe_allow_html=True)
st.markdown(header_text_5, unsafe_allow_html=True)


with st.form("input_form"):
    # Taking input review here
    # source: https://discuss.streamlit.io/t/change-font-size-and-font-color/12377/3
    # enter_review_here = '<p style="color:Black; font-size: 20px;">Upload numerical digit here</p>'
    # st.markdown(enter_review_here, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a file", type="jpg")
    st.markdown('')
    st.markdown('')
    # Predict digit button
    submitted = st.form_submit_button("Recognize digit")

if uploaded_file is None and submitted:
    preicting_text = '<p style="color:White; text-align:center; background-color:Red; font-size: 20px;">please upload an image contains digit!</p>'
    st.markdown(preicting_text, unsafe_allow_html=True)
    
else:
    if submitted:

        
        image = Image.open(uploaded_file)
        input_image = preprocess_image(image)
        
        upload_mssg = '<p style="color:Black; font-size: 10px;">Uploaded image</p>'
        st.markdown(upload_mssg, unsafe_allow_html=True)
        st.image(image, caption='', width=DIM[0])

        digit = predict_digit(input_image, model)

        output = f"Digit: {digit}"
        st.success(output)
    
    
hide_footer_style = """
<style>
.reportview-container .main footer {visibility: hidden; }    
"""
st.markdown(hide_footer_style, unsafe_allow_html=True)

created_by = '<p style="text-align:center; color:Black">Created by Shailesh</p>'
st.markdown(created_by, unsafe_allow_html=True)

feedback = '<p style="text-align:center; color:Black">Feel free to write your feedback @ shailvesteinsqrt@gmail.com</p>'
st.markdown(feedback, unsafe_allow_html=True)
