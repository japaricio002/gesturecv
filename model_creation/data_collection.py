import cv2
import os
import time
from datetime import datetime
import platform
import sys

class DataCollector:
    def __init__(self, base_path="dataset"):
        self.base_path = base_path
        self.gestures = ['thumbs_up', 'thumbs_down', 'peace', 'stop', 'okay']
        self.current_gesture = 0
        self.frame_count = 0
        self.recording = False
        self.setup_camera()
        
    def setup_camera(self):
        """Initialize camera based on platform"""
        system = platform.system()
        self.cap = None
        
        print("Attempting to initialize camera...")
        
        if system == "Darwin":  # macOS
            print("MacOS detected, using AVFoundation...")
            self.cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        else:  # Windows/Linux
            print("Windows/Linux detected, using default backend...")
            self.cap = cv2.VideoCapture(0)
        
        # Wait for camera to initialize
        time.sleep(2)
        
        if not self.cap.isOpened():
            print("Error: Could not open camera!")
            print("Please check if:")
            print("1. Your camera is connected")
            print("2. Your camera isn't being used by another application")
            print("3. You have camera permissions enabled")
            sys.exit(1)
        
        # Try to read a test frame
        ret, frame = self.cap.read()
        if not ret or frame is None:
            print("Error: Could not read frame from camera!")
            self.cap.release()
            sys.exit(1)
            
        print("Camera initialized successfully!")
    
    def create_directories(self):
        """Create necessary directories for dataset"""
        for gesture in self.gestures:
            path = os.path.join(self.base_path, gesture)
            os.makedirs(path, exist_ok=True)
            print(f"Created directory: {path}")
    
    def collect_data(self, frames_per_gesture=100):
        """Collect webcam frames for each gesture"""
        print("\nStarting data collection...")
        print("Controls:")
        print("- Press 'r' to start/stop recording frames")
        print("- Press 'n' to skip to next gesture")
        print("- Press 'q' to quit")
        print("\nWaiting for input...")
        
        self.create_directories()
        
        while self.current_gesture < len(self.gestures):
            gesture = self.gestures[self.current_gesture]
            
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print("Error: Failed to read frame")
                break
            
            # Create info display
            info_display = frame.copy()
            
            # Display current status
            cv2.putText(info_display, f"Current Gesture: {gesture}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(info_display, f"Progress: {self.frame_count}/{frames_per_gesture}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display recording status
            status = "Recording" if self.recording else "Paused"
            color = (0, 0, 255) if self.recording else (0, 255, 0)
            cv2.putText(info_display, f"Status: {status}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Display controls
            cv2.putText(info_display, "R: Record/Pause", (10, info_display.shape[0] - 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(info_display, "N: Next Gesture", (10, info_display.shape[0] - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(info_display, "Q: Quit", (10, info_display.shape[0] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow('Data Collection', info_display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('r'):
                self.recording = not self.recording
                print(f"Recording {'started' if self.recording else 'paused'}")
            
            elif key == ord('n'):
                self.current_gesture += 1
                self.frame_count = 0
                self.recording = False
                if self.current_gesture < len(self.gestures):
                    print(f"\nMoving to next gesture: {self.gestures[self.current_gesture]}")
                continue
            
            elif key == ord('q'):
                print("\nQuitting...")
                break
            
            if self.recording:
                # Save frame
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = os.path.join(self.base_path, gesture, 
                                      f"{gesture}_{timestamp}.jpg")
                cv2.imwrite(filename, frame)
                
                self.frame_count += 1
                if self.frame_count >= frames_per_gesture:
                    print(f"\nCompleted collecting {gesture}")
                    self.current_gesture += 1
                    self.frame_count = 0
                    self.recording = False
                    if self.current_gesture < len(self.gestures):
                        print(f"Moving to next gesture: {self.gestures[self.current_gesture]}")
                
                time.sleep(0.1)  # Delay between captures
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("\nData collection completed!")
        print(f"Data saved in: {os.path.abspath(self.base_path)}")

if __name__ == "__main__":
    try:
        collector = DataCollector()
        collector.collect_data()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
    finally:
        if 'collector' in locals() and hasattr(collector, 'cap'):
            collector.cap.release()
        cv2.destroyAllWindows()