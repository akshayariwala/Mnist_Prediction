import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize images
X_test = X_test / 255.0

# Reshape for CNN input
X_test = X_test.reshape(-1, 28, 28, 1)

# Load trained model
mnist_model = tf.keras.models.load_model('mnist_model.h5')

# Predict probabilities
predicted_probs = mnist_model.predict(X_test)

# Get predicted classes
predicted_classes = np.argmax(predicted_probs, axis=1)

# Find correct and incorrect predictions
correct_indices = np.where(predicted_classes == y_test)[0]
incorrect_indices = np.where(predicted_classes != y_test)[0]

# Streamlit UI
st.title("MNIST Digit Classifier Evaluation")

st.write(f"{len(correct_indices)} classified correctly")
st.write(f"{len(incorrect_indices)} classified incorrectly")

# ---------------- CORRECT PREDICTIONS ----------------
st.header("Correct Predictions")

correct_columns = st.columns(3)

for i, idx in enumerate(correct_indices[:9]):
    fig, ax = plt.subplots()

    ax.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    ax.set_title(f"Predicted: {predicted_classes[idx]} | Truth: {y_test[idx]}")
    ax.axis("off")

    with correct_columns[i % 3]:
        st.pyplot(fig)

# ---------------- INCORRECT PREDICTIONS ----------------
st.header("Incorrect Predictions")

incorrect_columns = st.columns(3)

for i, idx in enumerate(incorrect_indices[:9]):
    fig, ax = plt.subplots()

    ax.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    ax.set_title(f"Predicted: {predicted_classes[idx]} | Truth: {y_test[idx]}")
    ax.axis("off")

    with incorrect_columns[i % 3]:
        st.pyplot(fig)  