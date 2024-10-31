import torch
import cv2
import numpy as np
from pathlib import Path
import torchvision.transforms as transforms
from PIL import Image
import time

def load_model():
    """Load YOLOv5 model"""
    model = torch.hub.load('ultralytics/yolov5', 'custom', 'path/to/your/weights.pt')
    model.eval()
    return model

def preprocess_frame(frame):
    """Preprocess frame for model input"""
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert to PIL Image
    image = Image.fromarray(frame_rgb)
    
    # Apply transformations
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])
    
    return transform(image).unsqueeze(0)

def detect_gestures(model, frame):
    """Detect hand gestures in frame"""
    # Preprocess frame
    input_tensor = preprocess_frame(frame)
    
    # Run inference
    with torch.no_grad():
        predictions = model(input_tensor)
    
    return predictions.xyxy[0].cpu().numpy()

def draw_detections(frame, detections, gesture_classes):
    """Draw bounding boxes and labels on frame"""
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        if conf > 0.5:  # Confidence threshold
            # Convert coordinates to integers
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            gesture_name = gesture_classes[int(cls_id)]
            label = f"{gesture_name}: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Load model
    model = load_model()
    
    # Define gesture classes (update based on your trained model)
    gesture_classes = ['thumbs_up', 'thumbs_down', 'peace', 'stop', 'okay']
    
    # FPS calculation variables
    prev_time = time.time()
    fps = 0
    
    print("Press 'q' to quit")
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Detect gestures
        detections = detect_gestures(model, frame)
        
        # Draw detections on frame
        frame = draw_detections(frame, detections, gesture_classes)
        
        # Calculate and display FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow('Hand Gesture Recognition', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()