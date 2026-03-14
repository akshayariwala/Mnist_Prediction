# Mnist_Prediction

Handwritten Digit Recognition using CNN

Project Overview This project is a Deep Learning Web Application that recognizes handwritten digits (0–9) using a Convolutional Neural Network(CNN) trained on the MNIST dataset. The application allows users to upload an image of a handwritten digit and the trained model predicts the digit.

Technologies Used - Python - TensorFlow / Keras - Streamlit - NumPy -Pillow (PIL)

Project Structure

MNIST-Digit-Prediction │ ├── mnist_model.h5 ├── mnist_streamlit.py ├── requirements.txt └── README.txt

Dataset The model is trained using the MNIST dataset which contains: - 60,000 training images - 10,000 testing images - Image size: 28x28 pixels - Grayscale handwritten digits (0–9)

Requirements streamlit tensorflow numpy pillow matplotlib

Running the Application

Run the Streamlit app: streamlit run mnist_streamlit.py

Open your browser and go to: http://localhost:8501

How It Works 
1. User uploads an image containing a handwritten digit. 
2.Image is resized to 28x28 pixels. 
3. Image is converted to grayscale. 
4.Pixel values are normalized. 
5. The CNN model predicts the digit. 
6. The predicted result is displayed in the Streamlit interface.

Future Improvements - Add digit drawing canvas - Show prediction probability graph - Improve UI design - Deploy on Streamlit Cloud
