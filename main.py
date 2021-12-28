import numpy as np
from PIL import Image
from PIL import *
import tensorflow as tf
from tensorflow import keras
import streamlit as st

def image_resize(image, width = None, height = None):
    # resize the image
#     resized = cv2.resize(image, dim, interpolation = inter)
#     resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
#     # return the resized image
#     return np.array(resized)
    pass



def load_model():
    seq_model = keras.Sequential([
                                  keras.layers.Dense(500, input_shape=(784,), activation='relu'),
                                  keras.layers.Dense(100, activation='relu'),
                                  keras.layers.Dense(10, activation='sigmoid')
                                ])
    seq_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
    seq_model.load_weights('./digit_recognizer_model.h5')
    return seq_model


def preprocess_image(image):
    resized_image = image_resize(image, width=28, height=28)
    normalized_image = resized_image/255
    normalized_image = normalized_image.reshape(1,-1)
    return normalized_image


def predict_digit(normalized_image, model):
    result = model.predict(normalized_image)
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
# header_text_1 = '<p style="color:Black; font-size: 20px;">1.  This is AI based web-app, it uses support-vector-machine algorithm in backend to find polarity score.</p>'
header_text_1 = '<p style="color:Black; font-size: 20px;">1.  This is AI based web-app.</p>'
header_text_2 = '<p style="color:Black; font-size: 20px;">2.  It tells which numerical digit is in the picture.</p>' # This app is 86% confident that given review is good/bad.</p>'
st.markdown(header_text_1, unsafe_allow_html=True)
st.markdown(header_text_2, unsafe_allow_html=True)



with st.form("input_form"):
    # Taking input review here
    # source: https://discuss.streamlit.io/t/change-font-size-and-font-color/12377/3
    # enter_review_here = '<p style="color:Black; font-size: 20px;">Upload numerical digit here</p>'
    # st.markdown(enter_review_here, unsafe_allow_html=True)
    upload = st.file_uploader("Choose a file", type="jpg")
    st.markdown('')
    st.markdown('')
    # Predict digit button
    submitted = st.form_submit_button("Recognize digit")

# image = Image.open(uploaded_file)


# if submitted:
#     # If user dosen't enter any word/sentence and press predict polarity than show this message
#     if review == '' or review == 'write your review here':
#         review_error = '<p style="color:Orange; text-align:center; background-color:Blue; font-size: 20px;">**Please enter your review!**</p>'
#         st.markdown(review_error, unsafe_allow_html=True)
        
#     # It will show polarity of review
#     else:
#         #positive_review = '<p style="color:Green; text-align:center; font-size: 20px;">Positive review</p>'
#         #negative_review = '<p style="color:Red; text-align:center; font-size: 20px;">Negative review</p>'
#         positive_review = '<p style="color:White; text-align:center; background-color:Green; font-size: 20px;">Positive</p>'
#         negative_review = '<p style="color:White; text-align:center; background-color:Red; font-size: 20px;">Negative</p>'
        
#         if polarity == 1:
#             st.markdown(positive_review, unsafe_allow_html=True)
#         else:
#             st.markdown(negative_review, unsafe_allow_html=True)

            
if submitted:
#     image = np.array(upload)
    st.image(upload, caption='Sunrise by the mountains')
    
hide_footer_style = """
<style>
.reportview-container .main footer {visibility: hidden; }    
"""
st.markdown(hide_footer_style, unsafe_allow_html=True)

created_by = '<p style="text-align:center; color:Black">Created by Shailesh</p>'
st.markdown(created_by, unsafe_allow_html=True)

feedback = '<p style="text-align:center; color:Black">Feel free to write your feedback @ shailvesteinsqrt@gmail.com</p>'
st.markdown(feedback, unsafe_allow_html=True)
