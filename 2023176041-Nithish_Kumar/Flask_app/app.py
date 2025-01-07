import numpy as np
import os
from PIL import Image
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename, redirect
from gevent.pywsgi import WSGIServer
from keras.preprocessing import image
from flask import send_from_directory
import cv2
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
import pickle
 

UPLOAD_FOLDER = 'uploads'


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


model = tf.keras.models.load_model("models/xception_model_50.h5")
org_model = tf.keras.models.load_model("models/Xception_50.h5")
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == "POST":
        ans = ""
        org_ans = ""  # Initialize for org_model final output
        f = request.files["video"]  # Change to video upload
        filepath = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filepath))

        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filepath)
        key_frames = extract_key_frames(video_path)

        predictions = []
        org_predictions = []
        frame_paths = []
        frame_results = []  # Store all details for HTML rendering

        for i, frame in enumerate(key_frames):
            # Save frame as an image file
            frame_filename = f"frame_{i}.jpg"
            frame_path = os.path.join("static/frames", frame_filename)
            cv2.imwrite(frame_path, frame)  # Save frame as an image
            frame_paths.append(frame_path)  # Store the path for HTML rendering

            # Preprocess frame
            processed_frame = preprocess_frame(frame)

            # Predictions from the first model
            pred = model.predict(processed_frame)
            num = np.argmax(pred, axis=1)
            model_percentage = pred[0][num[0]]
            model_label = 'fake' if num[0] == 1 else 'real'

            # Predictions from org_model
            org_pred = org_model.predict(processed_frame)
            org_num = np.argmax(org_pred, axis=1)
            org_model_percentage = org_pred[0][org_num[0]]
            org_model_label = 'fake' if org_num[0] == 1 else 'real'

            # Append to results
            frame_results.append({
                "frame_no": i + 1,
                "frame_path": frame_path,
                "org_model_percentage": org_model_percentage * 100,
                "org_model_label": org_model_label,
                "model_percentage": model_percentage * 100,
                "model_label": model_label,
            })

            predictions.append(num[0])
            org_predictions.append(org_num[0])

        # Final prediction for the main model (majority vote)
        count_fake = predictions.count(1)
        count_real = predictions.count(0)
        ans = "FAKE" if count_fake >= count_real else "REAL"

        # Final prediction for org_model (majority vote)
        org_count_fake = org_predictions.count(1)
        org_count_real = org_predictions.count(0)
        org_ans = "FAKE" if org_count_fake >= org_count_real else "REAL"

        # Pass results to the template
        return render_template('predict.html', frame_results=frame_results, ans=ans, org_ans=org_ans)


#-----------------------------------------------------------------------------------------------------#

def extract_key_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    key_frames = []

    last_frame = None
    diffs = []

    # Calculate differences between consecutive frames
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if last_frame is not None:
            # Compute the difference between the current frame and the last frame
            diff = cv2.absdiff(frame, last_frame)
            diff_sum = np.sum(diff)
            diffs.append((diff_sum, i))
        last_frame = frame

    # Sort frames by the largest differences and select key frames
    diffs.sort(reverse=True, key=lambda x: x[0])
    key_frame_indices = [index for _, index in diffs[:10]]
    key_frame_indices.sort()  # Sort in order of appearance

    # Extract key frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start
    for i in range(total_frames):
        ret, frame = cap.read()
        if i in key_frame_indices and ret:
            key_frames.append(frame)
    
    cap.release()
    return key_frames

def preprocess_frame(frame):
    # Resize and preprocess the frame
    frame = cv2.resize(frame, (224, 224))  # Resize to (224, 224)
    frame = frame.astype('float32') / 255.0  # Normalize to [0, 1]
    return np.expand_dims(frame, axis=0)  # Add batch dimension

#--------------------------------------------------------------------------------------------------------#


if __name__ == '__main__':
    app.run(debug=True)
