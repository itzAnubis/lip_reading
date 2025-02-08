import cv2
import dlib
import numpy as np
import os

class VideoPreprocessor:
    def __init__(self, face_predictor_path = 'face_predictor/shape_predictor_68_face_landmarks.dat'):
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor(face_predictor_path)
    
    def extract_mouth_region(self, frame):
        # Detect face
        faces = self.face_detector(frame)
        if len(faces) == 0:
            return None
        
        # Get facial landmarks
        landmarks = self.landmark_predictor(frame, faces[0])
        
        # Extract mouth coordinates (points 48-68 in dlib's facial landmarks)
        mouth_points = np.array([[landmarks.part(i).x, landmarks.part(i).y] 
                                for i in range(48, 68)])
        
        # Get bounding box
        x, y = np.min(mouth_points, axis=0)
        w, h = np.max(mouth_points, axis=0) - np.min(mouth_points, axis=0)
        
        # Add padding
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w += 2 * padding
        h += 2 * padding
        
        # Extract and resize mouth region
        mouth_region = frame[y:y+h, x:x+w]
        mouth_region = cv2.resize(mouth_region, (64, 64))
        
        return mouth_region
    
    def preprocess_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            mouth_region = self.extract_mouth_region(frame)
            if mouth_region is not None:
                mouth_region = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)
                frames.append(mouth_region)
        
        cap.release()
        
        # Ensure consistent sequence length
        if len(frames) > 75:
            frames = frames[:75]
        else:
            # Pad with zeros if needed
            padding = [np.zeros_like(frames[0]) for _ in range(75 - len(frames))]
            frames.extend(padding)
            
        return np.array(frames)
    
    def save_frames(self, video_path, transcription_path, save_path='data/processed'):
        # Extract frames
        frames = self.preprocess_video(video_path)
        total_frames = len(frames)
        
        # Read transcription file
        word_segments = []
        with open(transcription_path, 'r') as f:
            for line in f:
                start, end, word = line.strip().split()
                word_segments.append({
                    'word': word,
                    'start': int(start),
                    'end': int(end)
                })
        
        # Calculate total duration
        total_duration = word_segments[-1]['end'] - word_segments[0]['start']
        
        # Create base directory
        file_name = os.path.basename(video_path).split('.')[0]
        base_path = os.path.join(save_path, file_name)
        
        # Create word folders and distribute frames
        for segment in word_segments:
            # Create folder for this word
            word_path = os.path.join(base_path, segment['word'])
            os.makedirs(word_path, exist_ok=True)
            
            # Calculate frame range for this word
            segment_duration = segment['end'] - segment['start']
            segment_proportion = segment_duration / total_duration
            
            start_frame = int((segment['start'] - word_segments[0]['start']) / total_duration * total_frames)
            end_frame = int((segment['end'] - word_segments[0]['start']) / total_duration * total_frames)
            
            # Save frames for this segment
            for i in range(start_frame, end_frame):
                if i < len(frames):
                    img = cv2.resize(frames[i], (256, 256))
                    save_path = os.path.join(word_path, f'frame_{i}.jpg')
                    cv2.imwrite(save_path, img)
                    # print(f"Saving frame {i} to word: {segment['word']}")  # Debug print
        
    def count_frames(video_path):
        """Counts the number of frames in a video.

        Args:
            video_path (str): Path to the video file.

        Returns:
            int: Number of frames in the video.
        """

        cap = cv2.VideoCapture(video_path)

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cap.release()

        return frame_count