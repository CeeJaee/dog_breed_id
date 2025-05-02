import os
import random
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('dogs.csv')

# Load the VGG16 model pre-trained on ImageNet
model = VGG16(weights='imagenet')

# Function to predict the class of an image
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    return decoded_predictions

# Function to display the image and predictions
def display_image_and_predictions(img_path, predictions):
    img = Image.open(img_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title("Predictions:")
    for i, (imagenet_id, label, score) in enumerate(predictions):
        plt.text(0, -10*(i+1), f"{i+1}: {label} ({score:.2f})", fontsize=12, color='red')
    plt.show()

# Main loop to accept commands
while True:
    command = input("Enter '1' to load a new image or 'q' to quit: ")
    if command == '1':
        img_path = random.choice(df['filepaths'])
        print(f"Loading image: {img_path}")
        predictions = predict_image(img_path)
        display_image_and_predictions(img_path, predictions)
    elif command == 'q':
        print("Exiting program.")
        break
    else:
        print("Invalid command. Please enter '1' to load a new image or 'q' to quit.")
