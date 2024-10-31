import cv2
import torch
import numpy as np
import pygame
from pathlib import Path

class GestureControlApp:
    def __init__(self, model_path='best.pt'):
        # Initialize model
        self.model = self.load_model(model_path)
        self.gesture_classes = ['thumbs_up', 'thumbs_down', 'peace', 'stop', 'okay']
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Initialize display window
        cv2.namedWindow('Gesture Control', cv2.WINDOW_NORMAL)
        
        # Initialize shape position
        self.shape_position = 240  # Middle of 480 height
        self.shape_size = 50
        
    def load_model(self, model_path):
        """Load YOLOv5 model"""
        try:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
            model.eval()
            # Move to GPU if available
            if torch.cuda.is_available():
                model.cuda()
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def process_frame(self, frame):
        """Process frame through YOLO model"""
        # Convert frame to RGB (YOLO expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run inference
        results = self.model(frame_rgb)
        
        # Get predictions
        predictions = results.xyxy[0].cpu().numpy()
        return predictions
    
    def update_shape_position(self, gesture):
        """Update shape position based on gesture"""
        move_speed = 10
        if gesture == 'thumbs_up':
            self.shape_position = max(self.shape_size//2, self.shape_position - move_speed)
        elif gesture == 'peace':
            self.shape_position = min(480 - self.shape_size//2, self.shape_position + move_speed)
    
    def draw_frame(self, frame, predictions):
        """Draw predictions and shape on frame"""
        # Draw detections
        for pred in predictions:
            x1, y1, x2, y2, conf, cls_id = pred
            if conf > 0.5:  # Confidence threshold
                # Draw bounding box
                cv2.rectangle(frame, 
                            (int(x1), int(y1)), 
                            (int(x2), int(y2)), 
                            (0, 255, 0), 2)
                
                # Draw label
                cls_name = self.gesture_classes[int(cls_id)]
                label = f"{cls_name}: {conf:.2f}"
                cv2.putText(frame, label, 
                           (int(x1), int(y1) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 0), 2)
                
                # Update shape position based on gesture
                self.update_shape_position(cls_name)
        
        # Draw moving shape on right side of frame
        shape_x = frame.shape[1] - 100  # 100 pixels from right edge
        cv2.circle(frame, 
                  (shape_x, int(self.shape_position)),
                  self.shape_size//2,
                  (0, 0, 255),
                  -1)
        
        # Draw instructions
        instructions = [
            "Controls:",
            "Open Palm - Move up",
            "peace - Move down",
            "Press 'q' to quit"
        ]
        for i, text in enumerate(instructions):
            cv2.putText(frame, text,
                       (10, 30 + i*30),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (255, 255, 255), 2)
        
        return frame
    
    def run(self):
        """Main application loop"""
        print("Starting Gesture Control App...")
        print("Controls:")
        print("- Show thumbs up to move shape up")
        print("- Show thumbs down to move shape down")
        print("- Press 'q' to quit")
        
        try:
            while True:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Error reading from webcam")
                    break
                
                # Mirror frame horizontally
                frame = cv2.flip(frame, 1)
                
                # Process frame
                predictions = self.process_frame(frame)
                
                # Draw frame
                display_frame = self.draw_frame(frame, predictions)
                
                # Show frame
                cv2.imshow('Gesture Control', display_frame)
                
                # Break loop on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except Exception as e:
            print(f"Error during execution: {e}")
        
        finally:
            # Cleanup
            self.cap.release()
            cv2.destroyAllWindows()

def main():
    try:
        # Create and run app
        app = GestureControlApp()
        app.run()
    except Exception as e:
        print(f"Application error: {e}")

if __name__ == "__main__":
    main()