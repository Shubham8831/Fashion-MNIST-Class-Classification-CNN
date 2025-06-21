import streamlit as st  # Import Streamlit for building the web app interface
import numpy as np  # Import NumPy for numerical operations on arrays
from PIL import Image, ImageOps  # Import PIL for image loading and processing
from tensorflow import keras  # Import TensorFlow Keras API for model handling
from keras.models import load_model  # Import function to load a saved Keras model

# ——————————————————————————————
# 1. Load model & class names
# ——————————————————————————————
model = load_model("fashion_mnist_model.h5")  # Load the trained Fashion MNIST CNN from disk
class_names = [  # Define the list of human-readable labels for each class
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]



# ——————————————————————————————
# 2. Sidebar: Project Details
# ——————————————————————————————
with st.sidebar.expander("📄 Project Details", expanded=True):  # Create an expandable section in the sidebar
    st.markdown("""  
    **#1. Problem:**  
    Classify clothing images into 10 categories.
    
    **#2. Data:**    
    Fashion MNIST (70k 28×28 grayscale images).
    
    **#3. Model:**       
    2×Conv2D + MaxPooling, Dropout, Dense(128), Softmax(10).
    
    **#4. Training :**   Adam optimizer
    
    **#5. App Workflow:**   
    Upload image → preprocess → predict → visualize results.
    
    **#6. Next Steps:**   
    Add data augmentation, deploy on Streamlit Cloud or Docker.
    
    """)  # End of Markdown
    st.markdown("---")  # Horizontal separator
    st.markdown("Made with ❤️ using TensorFlow & Streamlit by Shubham")  # Footer




# ——————————————————————————————
# 3. Main Title & Hero
# ——————————————————————————————
st.markdown(  # Render the main title and subtitle with custom styling
    """
    <h1 style=\"text-align:center; color:#4B4BFF;\">👗 Fashion MNIST Classifier</h1>
    <p style=\"text-align:center; color:#555;\">Upload any clothing image and see our CNN’s prediction!</p>
    """,
    unsafe_allow_html=True  # Allow HTML/CSS in Markdown
)




# ——————————————————————————————
# 4. Preprocessing Function
# ——————————————————————————————
def preprocess_uploaded_image(image_file):  # function to prepare uploaded images for prediction
    image = Image.open(image_file).convert('L')  #  convert to grayscale
    image = ImageOps.invert(image)  # Invert image colors (white on black) for consistency
    image = image.resize((28, 28))  # Resize image to 28×28 pixels as the model expects
    arr = np.array(image) / 255.0  # Normalize pixel values to the range [0, 1]
    return arr.reshape(1, 28, 28, 1)  # Add batch and channel dimensions for model input




# ——————————————————————————————
# 5. Two-Column Layout: Upload vs Predict
# ——————————————————————————————
col1, col2 = st.columns([1, 1], gap='large')  # Create two columns 



with col1:  # In the left column:
    st.header("Upload & Preview")  # Section header for upload
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])  # File uploader
    if uploaded_file:  # If an image is uploaded:
        img_arr = preprocess_uploaded_image(uploaded_file)  # Preprocess the uploaded image
        st.image(img_arr.squeeze(), caption="28×28 Grayscale", width=300)  # Display the processed 28×28 image



with col2:  # In the right column:
    st.header("Prediction")  
    if uploaded_file and st.button("🔍 Predict"):  # When Predict button is clicked:
        
        preds = model.predict(img_arr)[0]  # get prediction probabilities

        #chatgpt se churaya huu
        top_idx = np.argmax(preds)  # Identify the class with highest probability
        st.metric("Predicted Class", class_names[top_idx], delta=f"{preds[top_idx]*100:.1f}%")  # Display predicted class and confidence
        st.bar_chart({class_names[i]: float(preds[i]) for i in range(10)})  # Plot confidence scores for all classes



# ——————————————————————————————
# 6. Footer 
# ——————————————————————————————
st.markdown("---")  # Separator before the custom footer
hide_streamlit_style = """
    <style>
      #MainMenu {visibility: hidden;}  # Hide the default Streamlit menu
      footer {visibility: hidden;}    # Hide the default footer
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)  # Apply custom CSS to hide Streamlit elements
