# app.py
from flask import Flask, render_template, request, redirect, url_for, session
import os
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Load the trained model
model = load_model('trained_model1.h5')
model1=load_model('chest_xray_cnn_model.h5')
def predict_image(image_path):
    img = image.load_img(image_path, target_size=(150, 150), color_mode="rgb")
    img_array = image.img_to_array(img)
    img_array /= 255.0
    img_array = np.expand_dims(img_array, axis=0)
    # Predict the class
    prediction = model1.predict(img_array)
    if prediction[0] < 0.5:
        return "NON-CHEST X-RAY"
    else:
        return "CHEST X-RAY"

# Function to predict the class of the image
def predict_image_class(image_path):
    img = image.load_img(image_path, target_size=(150, 150), color_mode="grayscale")
    img_array = image.img_to_array(img)
    img_array /= 255.0
    img_array = np.expand_dims(img_array, axis=0)
    labels = ['PNEUMONIA', 'NORMAL', 'TUBERCULOSIS', 'COVID19']
    # Predict the class
    prediction = model.predict(img_array)
    predicted_class = labels[np.argmax(prediction)]
    return predicted_class

# Database initialization
def init_db():
    conn = sqlite3.connect('database.db')
    print("Opened database successfully")

    conn.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT, password TEXT)')
    print("Table created successfully")
    conn.close()

init_db()



# Signup route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        hashed_password = generate_password_hash(password)

        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (email, password) VALUES (?, ?)", (email, hashed_password))
        conn.commit()
        conn.close()

        return redirect(url_for('login'))
    return render_template('signup.html')

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email=?", (email,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user[2], password):
            session['logged_in'] = True
            session['email'] = email
            return redirect(url_for('index'))
        else:
            return 'Invalid email or password. Please try again.'
    return render_template('login.html')

# Index route
@app.route('/index')
def index():
    if 'logged_in' in session:
        return render_template('index.html')
    return redirect(url_for('login'))

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if 'logged_in' in session:
            image_file = request.files['image']
            if image_file:
                # Save the uploaded image
                image_path = os.path.join('static', 'uploaded_image.jpg')
                image_file.save(image_path)
                # Check if the image is a chest X-ray
                chest_xray = predict_image(image_path)
                if chest_xray == "CHEST X-RAY":
                    # If it's a chest X-ray, predict the class
                    predicted_class = predict_image_class(image_path)
                    return render_template('index.html', prediction=predicted_class, image_path=image_path)
                else:
                    # If it's not a chest X-ray, return "No X-ray image"
                    return render_template('index.html', prediction="Not an X-ray image", image_path=image_path)
            return 'No image uploaded.'
        return redirect(url_for('login'))
    return redirect(url_for('index'))


# Home route
@app.route('/')
def home():
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)