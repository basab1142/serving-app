import streamlit as st
from model_build import *
from PIL import Image
st.sidebar.header('Other model to play with')
st.sidebar.markdown("""
       * [Dog Classification](www.google.com)
       * [Sentiment Analysis](www.google.com)             
""")

col1, ex1, col2, ex2,col3 = st.columns([3,1,3,1,2])


col1.markdown("# Hello & Welcome to the page")

col1.image('example.jpeg')

col1.write("Below picture depicts how the model works")
col1.write("## MODEL OVERVIEW")
col1.image('model_pic.png')

uploaded_photo = col2.file_uploader("Upload a photo")

PHOTO = '1.jpeg'

camera_photo = col2.camera_input("Take a photo and generate caption")
caption = ""
if uploaded_photo:
    image_path = Image.open(uploaded_photo)  
    caption = get_caption_2(model, image_processor, tokenizer, image_path)
elif camera_photo:
    image_path = Image.open(camera_photo)  
    caption = get_caption_2(model, image_processor, tokenizer, image_path)

col3.markdown(f"""
    ## Output of the image

    {caption}        



   
""")    
