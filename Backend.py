from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import numpy as np
import cv2
import pickle
import preprocessor

import keras



with open('/mnt/g/projects/lip_reading_project/finall_app_ISA/word_index.pkl', 'rb') as f:
    class_mapping = pickle.load(f)

inverse_word_index = {index: word for word, index in class_mapping.items()}



sc = preprocessor.VideoPreprocessor()

# Initialize Flask app
app = Flask(__name__)
CORS(app)



# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



def prossec(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        mouth_region = sc.extract_mouth_region(frame)
        if mouth_region is not None:
            mouth_region = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)
            frames.append(mouth_region)
    
    cap.release()
    frames = np.asarray(frames)
    
    frames = np.expand_dims(frames, axis=-1)
    frames = np.expand_dims(frames, axis=0)
    return frames

def predict(preprocessed_frames):
    predictions = model.predict(preprocessed_frames)

    # Extract indices from predictions
    predicted_indices = np.argmax(predictions, axis=-1).flatten()
    # Map indices to words
    predicted_words = [inverse_word_index.get(index, "None ") for index in predicted_indices]

    sentence = " ".join(predicted_words)
    
    return sentence
# Load the pre-trained model
model = keras.models.load_model('/mnt/g/projects/lip_reading_project/finall_app_ISA/Sequantial_words_.keras')

# print(model.summary())

@app.route('/predict_uploaded_video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    print(f"Received file: {file.filename}, Content-Type: {file.content_type}")

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        preprocessed_frames = prossec(file_path)
        
        num_frames = preprocessed_frames.shape[1]
        print(f"Number of frames: {num_frames}")
        
        if num_frames > 75:
            chunks = []
            for start in range(0, num_frames, 75):
                chunk = preprocessed_frames[0, start:start+75]  
                if chunk.shape[0] < 75:
                    padding = np.zeros((75 - chunk.shape[0], chunk.shape[1], chunk.shape[2], chunk.shape[3]))
                    chunk = np.vstack((chunk, padding))  

                chunks.append(chunk)

            preprocessed_frames = np.array(chunks)

        print(preprocessed_frames.shape)
         
        sentence = predict(preprocessed_frames)
        
        print(sentence)

        return jsonify({"message": "Video processed successfully!",
                        "predicted_class": sentence}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
            
if __name__ == '__main__':
    app.run(debug=True, port=5050)