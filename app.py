from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import pandas as pd
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the foundation shades dataset
dataset = pd.read_csv('shades.csv')

# Configure the upload folder
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def skin_color_extraction(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 40, 80], dtype=np.uint8)
    upper_skin = np.array([25, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    skin = cv2.bitwise_and(image, image, mask=mask)
    return skin

def suggest_matching_foundation(user_skin_color):
    dataset_cleaned = dataset.dropna(subset=['H', 'S', 'V'], how='any').copy()
    dataset_cleaned['hue_diff'] = np.abs(dataset_cleaned['H'] - user_skin_color[0])
    dataset_cleaned['sat_diff'] = np.abs(dataset_cleaned['S'] - user_skin_color[1])
    dataset_cleaned['val_diff'] = np.abs(dataset_cleaned['V'] - user_skin_color[2])
    dataset_cleaned['distance'] = dataset_cleaned['hue_diff'] * 0.5 + dataset_cleaned['sat_diff'] * 0.3 + dataset_cleaned['val_diff'] * 0.2
    sorted_shades = dataset_cleaned.sort_values(by='distance')
    return sorted_shades.iloc[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        user_image = cv2.imread(file_path)
        skin_color = skin_color_extraction(user_image)
        avg_hue = np.mean(skin_color[:, :, 0])
        avg_saturation = np.mean(skin_color[:, :, 1])
        avg_value = np.mean(skin_color[:, :, 2])
        user_skin_color = [avg_hue, avg_saturation, avg_value]
        matching_foundation = suggest_matching_foundation(user_skin_color)

        return render_template('result.html', 
                               brand=matching_foundation['brand'], 
                               product=matching_foundation['product'], 
                               hex_code=matching_foundation['hex'],
                               image_url=url_for('static', filename='uploads/' + filename))

if __name__ == "__main__":
    app.run(debug=True)
