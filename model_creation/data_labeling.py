import os
import cv2
import numpy as np
from pathlib import Path
import mediapipe as mp

class GestureLabeler:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
    
    def detect_hand(self, image):
        """Detect hand landmarks and return bounding box"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            h, w, _ = image.shape
            landmarks = results.multi_hand_landmarks[0]
            
            # Get bounding box coordinates
            x_coords = [lm.x * w for lm in landmarks.landmark]
            y_coords = [lm.y * h for lm in landmarks.landmark]
            
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))
            
            # Add padding
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)
            
            return [x_min, y_min, x_max, y_max]
        
        return None

    def create_yolo_label(self, bbox, image_shape, class_id):
        """Convert bounding box to YOLO format"""
        h, w, _ = image_shape
        x_center = ((bbox[0] + bbox[2]) / 2) / w
        y_center = ((bbox[1] + bbox[3]) / 2) / h
        width = (bbox[2] - bbox[0]) / w
        height = (bbox[3] - bbox[1]) / h
        
        return f"{class_id} {x_center} {y_center} {width} {height}"

    def process_dataset(self, dataset_path):
        """Process all images and create YOLO labels"""
        gestures = os.listdir(dataset_path)
        
        for gesture_id, gesture in enumerate(gestures):
            gesture_path = os.path.join(dataset_path, gesture)
            if not os.path.isdir(gesture_path):
                continue
            
            print(f"Processing {gesture}...")
            
            for image_file in os.listdir(gesture_path):
                if not image_file.endswith(('.jpg', '.jpeg', '.png')):
                    continue
                
                image_path = os.path.join(gesture_path, image_file)
                image = cv2.imread(image_path)
                
                if image is None:
                    print(f"Could not read {image_path}")
                    continue
                
                bbox = self.detect_hand(image)
                if bbox:
                    # Create label file
                    label_path = os.path.splitext(image_path)[0] + '.txt'
                    label = self.create_yolo_label(bbox, image.shape, gesture_id)
                    
                    with open(label_path, 'w') as f:
                        f.write(label)
                else:
                    print(f"No hand detected in {image_path}")

if __name__ == "__main__":
    labeler = GestureLabeler()
    labeler.process_dataset("dataset")