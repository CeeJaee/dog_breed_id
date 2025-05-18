import os
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
from config import MODEL_SAVE_PATH, IMAGE_SIZE
from data_loader import get_class_names
import numpy as np

UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model
model = load_model('models/gpretrained_dog_breed_classifier3.keras')

# Class names in exact order used during training
class_names = get_class_names()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Preprocess the image for model prediction"""
    img = Image.open(image_path)
    img = img.resize(IMAGE_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    return img_array

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        # If user does not select file, browser submits empty file
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Preprocess and predict
            img_array = preprocess_image(filepath)
            predictions = model.predict(img_array)
            
            # Get top 5 predictions
            top5_indices = np.argsort(predictions[0])[-5:][::-1]
            top5_classes = [class_names[i] for i in top5_indices]
            top5_probs = [predictions[0][i] for i in top5_indices]
            
            top5_predictions = list(zip(top5_classes, top5_probs))
            
            return render_template('index.html', 
                                 filename=filename,
                                 predictions=top5_predictions)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)