import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os

file_id ="1Dtc6aopehnUtOW78tpaTXGoaABwFjBo0"
url ='https://drive.google.com/uc?id=1Dtc6aopehnUtOW78tpaTXGoaABwFjBo0'
model_path ="trained_plant_disease_model.keras"


if not os.path.exists(model_path):
    st.warning("Downloading model from Google Drive...")
    gdown.download(url, model_path, quiet=False)

# Check if model was downloaded correctly
if os.path.exists(model_path):
    file_size = os.path.getsize(model_path)
    st.success(f"Model downloaded successfully! File size: {file_size} bytes")
    
    if file_size < 89200000:  # If too small, it might be an HTML error file
        st.error("The downloaded file is too small. It might not be the correct model file.")
else:
    st.error("Model file was not found after download. Please check the URL or file permissions.")



def model_prediction(test_image):
    model = tf.keras.models.load_model(model_path)
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebar
st.sidebar.title("Plant Disease Detection System for Sustainable Agriculture")
app_mode = st.sidebar.selectbox("Select Page",["HOME","DISEASE RECOGNITION","PROJECT DETAILS"])
#app_mode = st.sidebar.selectbox("Select Page",["Home"," ","Disease Recognition","PROJECT DETAILS"])
  
main_content = st.empty()

# import Image from pillow to open images
from PIL import Image
img = Image.open("Disease.png")

# display image using streamlit
# width is used to set the width of an image
st.image(img)

#Main Page
if(app_mode=="HOME"):
    main_content.empty()
    st.markdown("<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: blue;'>👉 Select 'Disease Recognition' from the sidebar to get started! 👈</h3>", unsafe_allow_html=True)
    moving_link = """
        <style>
            .glow {
                font-size: 20px;
                color: #00FFFF;
                text-align: center;
                animation: glow-effect 1s infinite alternate;
            }
            
            @keyframes glow-effect {
                from {
                    text-shadow: 0 0 5px #00FFFF, 0 0 10px #00FFFF, 0 0 20px #00FFFF;
                }
                to {
                    text-shadow: 0 0 10px #00FFFF, 0 0 20px #00FFFF, 0 0 30px #00FFFF;
                }
            }
        </style>
        
        <marquee behavior="scroll" direction="left" scrollamount="5">
            <span class="glow">✨</span> 
            <a href="https://github.com/Rushikesav/Test-data/tree/main/3.Potato%20Leaf%20Disease%20Detection/dataset/Test" target="_blank" style="text-decoration: none; color: blue;">
                Click here to download the test data to test the model! 🌿
            </a> 
            <span class="glow">✨</span>
        </marquee>
        """
        
        
    st.markdown(moving_link, unsafe_allow_html=True)

#Prediction Page
elif(app_mode=="DISEASE RECOGNITION"):
    main_content.empty()
    st.header("Plant Disease Detection System for Sustainable Agriculture")
    st.markdown('This app detects the potato leaf disease with upto 93 percent accuracy')
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        if test_image:
            st.image(test_image,width=4,use_container_width=True)
        else:
            st.warning("Please upload an image first.")
        
    #Predict button
    if(st.button("Predict")):
        if test_image:
            st.snow()
            st.write("Our Prediction:")
            result_index = model_prediction(test_image)
            
            # Class labels
            class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
            st.success(f"Model is predicting: **{class_names[result_index]}**")
        else:
            st.warning("Please upload an image first.")

# Project Details Page
elif(app_mode == "PROJECT DETAILS"):
    main_content.empty()
    st.title("Project Details and Model Working")
    
    st.header("1. Introduction")
    st.markdown("""
    The **Plant Disease Detection System** is an AI-powered tool designed to detect diseases in potato leaves.
    Using machine learning techniques, particularly deep learning, the model achieves up to **93% accuracy** in identifying:
    - **Early Blight**
    - **Late Blight**
    - **Healthy leaves**
    """)
    
    st.header("2. How the Model Works")
    st.markdown("""
    The model uses a **Convolutional Neural Network (CNN)**, a type of deep learning algorithm particularly effective for image classification tasks.
    
    **Steps:**
    1. **Preprocessing**: The uploaded image is resized to **128x128 pixels** to match the input size expected by the model.
    2. **Model Prediction**:
       - The model predicts the likelihood of the image belonging to each disease category.
       - The category with the highest probability is selected as the prediction.
    3. **Output**: The predicted class (e.g., **Potato___Early_blight**) is displayed to the user.
    """)

    st.header("3. Technologies Used")
    st.markdown("""
    - **TensorFlow/Keras** for building and training the deep learning model.
    - **Streamlit** for creating an interactive web interface.
    - **Google Drive** integration using **gdown** for downloading the trained model.
    """)

    st.header("4. Future Enhancements")
    st.markdown("""
    - **Expand to Other Crops**: Extend the model to detect diseases in other crops like tomatoes, wheat, etc.
    - **Real-time Detection**: Integrate with drones or mobile devices for real-time disease detection.
    - **Data Collection**: Incorporate user feedback to improve model accuracy.
    """)

