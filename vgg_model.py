import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import pandas as pd
import random

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

    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        print(f"{i+1}: {label} ({score:.2f})")

# Randomly select five images from the CSV file
random_images = df['filepaths'].sample(n=5, random_state=1)

# Example usage with randomly selected images
for img_path in random_images:
    print(f"Predictions for {img_path}:")
    predict_image(img_path)
    print("\n")
