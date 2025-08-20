# app.py
from flask import Flask, render_template, request
import os
import uuid
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load the trained models
model = load_model('trained_model1.h5')
model1 = load_model('chest_xray_cnn_model.h5')

def predict_image(image_path):
    """Check if image is a chest X-ray"""
    img = image.load_img(image_path, target_size=(150, 150), color_mode="rgb")
    img_array = image.img_to_array(img)
    img_array /= 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model1.predict(img_array)
    if prediction[0] < 0.5:
        return "NON-CHEST X-RAY"
    else:
        return "CHEST X-RAY"

def predict_image_class(image_path):
    """Predict the class of the chest X-ray"""
    img = image.load_img(image_path, target_size=(150, 150), color_mode="grayscale")
    img_array = image.img_to_array(img)
    img_array /= 255.0
    img_array = np.expand_dims(img_array, axis=0)
    labels = ['PNEUMONIA', 'NORMAL', 'TUBERCULOSIS', 'COVID19']
    prediction = model.predict(img_array)
    predicted_class = labels[np.argmax(prediction)]
    return predicted_class

# Main route - serves the X-ray classification page
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.files['image']
    if image_file and image_file.filename:
        # Validate file type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
        if '.' in image_file.filename and image_file.filename.rsplit('.', 1)[1].lower() in allowed_extensions:
            # Save the uploaded image with unique name
            filename = f"{uuid.uuid4().hex}.jpg"
            image_path = os.path.join('static', filename)
            image_file.save(image_path)
            
            try:
                # Check if the image is a chest X-ray
                chest_xray = predict_image(image_path)
                if chest_xray == "CHEST X-RAY":
                    # If it's a chest X-ray, predict the class
                    predicted_class = predict_image_class(image_path)
                    return render_template('index.html', prediction=predicted_class, image_path=image_path)
                else:
                    # If it's not a chest X-ray, return "No X-ray image"
                    return render_template('index.html', prediction="Not an X-ray image", image_path=image_path)
            except Exception as e:
                return render_template('index.html', prediction=f"Error processing image: {str(e)}")
        else:
            return render_template('index.html', prediction="Invalid file type. Please upload an image file.")
    else:
        return render_template('index.html', prediction="No file selected.")

if __name__ == '__main__':
    app.run(debug=True)