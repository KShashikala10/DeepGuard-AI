import os
from flask import Flask, request, render_template, redirect, url_for
import cv2
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
from mtcnn import MTCNN

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('temp', exist_ok=True)

# Model path
MODEL_PATH = 'models/deepfake_keras3_compatible.keras'

# Allowed extensions
IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Initialize MTCNN detector
detector = MTCNN()

# Image processing settings
IMG_SIZE = (224, 224)

def allowed_image_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in IMAGE_EXTENSIONS

def allowed_video_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in VIDEO_EXTENSIONS

def prepare_image(img_path):
    """Prepare image for prediction"""
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
    x = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    return np.expand_dims(x, axis=0)

def preprocess_face(face_img, target_size):
    """Preprocess face for video prediction"""
    face = cv2.resize(face_img, target_size)
    face = face.astype("float32") / 255.0
    return np.expand_dims(face, axis=0)

@app.route('/')
def index():
    """Main page with navigation"""
    return render_template('index.html')

@app.route('/image', methods=['GET', 'POST'])
def image_detector():
    """Image deepfake detection"""
    result = None
    img_filename = None
    error = None
    
    try:
        if request.method == 'POST':
            if 'file' not in request.files:
                error = 'No image file provided.'
                return render_template('image.html', error=error)
            
            file = request.files['file']
            
            if file.filename == '':
                error = 'No selected file.'
                return render_template('image.html', error=error)
            
            if not allowed_image_file(file.filename):
                error = 'Unsupported image file extension. Please use PNG, JPG, JPEG, GIF, or BMP.'
                return render_template('image.html', error=error)
            
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Prepare and predict
            x = prepare_image(filepath)
            pred = model.predict(x)[0][0]
            label = 'Real' if pred > 0.5 else 'Deepfake'
            confidence = float(pred if pred > 0.5 else 1 - pred)
            
            result = {
                'label': label,
                'confidence': f"{confidence*100:.2f}%"
            }
            
            img_filename = filename
    
    except Exception as e:
        error = f'Internal error: {str(e)}'
    
    return render_template('image.html', result=result, img_filename=img_filename, error=error)

@app.route('/video', methods=['GET', 'POST'])
def video_detector():
    """Video deepfake detection"""
    result = None
    confidence = None
    fake_votes = None
    real_votes = None
    error = None
    
    try:
        if request.method == 'POST':
            if 'video' not in request.files:
                error = 'No video file provided.'
                return render_template('video.html', error=error)
            
            file = request.files['video']
            
            if file.filename == '':
                error = 'No selected file.'
                return render_template('video.html', error=error)
            
            if not allowed_video_file(file.filename):
                error = 'Unsupported video file extension. Please use MP4, AVI, MOV, or MKV.'
                return render_template('video.html', error=error)
            
            filename = secure_filename(file.filename)
            temp_dir = 'temp'
            temp_path = os.path.join(temp_dir, filename)
            file.save(temp_path)
            
            # Process video
            cap = cv2.VideoCapture(temp_path)
            
            if not cap.isOpened():
                os.remove(temp_path)
                error = 'Failed to open video file.'
                return render_template('video.html', error=error)
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(round(fps / 2.0))
            if frame_interval == 0:
                frame_interval = 1
            
            frame_idx = 0
            fake_votes = 0
            real_votes = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % frame_interval == 0:
                    # Convert BGR to RGB for MTCNN
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Detect faces using MTCNN
                    faces = detector.detect_faces(rgb_frame)
                    
                    if len(faces) == 0:
                        frame_idx += 1
                        continue
                    
                    # Use the first detected face with highest confidence
                    face = faces[0]
                    x, y, w, h = face['box']
                    
                    # Ensure coordinates are within frame bounds
                    x, y = max(0, x), max(0, y)
                    x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
                    
                    if x2 <= x or y2 <= y:
                        frame_idx += 1
                        continue
                    
                    # Extract face region
                    face_img = frame[y:y2, x:x2]
                    
                    if face_img.size == 0:
                        frame_idx += 1
                        continue
                    
                    # Predict
                    inp = preprocess_face(face_img, target_size=model.input_shape[1:3])
                    pred = model.predict(inp, verbose=0)[0][0]
                    
                    if pred > 0.5:
                        real_votes += 1
                    else:
                        fake_votes += 1
                
                frame_idx += 1
            
            cap.release()
            os.remove(temp_path)
            
            total_votes = fake_votes + real_votes
            if total_votes == 0:
                error = 'No faces detected in video frames.'
                return render_template('video.html', error=error)
            
            result = "Real" if real_votes > fake_votes else "Fake"
            confidence_value = (real_votes if real_votes > fake_votes else fake_votes) / total_votes
            confidence = f"{confidence_value*100:.2f}%"
    
    except Exception as e:
        error = f'Internal error: {str(e)}'
    
    return render_template(
        'video.html',
        result=result,
        confidence=confidence,
        fake_votes=fake_votes,
        real_votes=real_votes,
        error=error
    )

@app.route('/reports')
def reports():
    """Reports page to view and manage analysis reports"""
    return render_template('reports.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
