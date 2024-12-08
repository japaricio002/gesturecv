# README.md

## Gesture Recognition and Control System

This repository is a comprehensive system for gesture recognition and application control, built using Python, OpenCV, and YOLOv5. The system allows for real-time gesture detection and mapping of gestures to specific actions, such as controlling an on-screen object. It is ideal for exploring computer vision, machine learning, and interactive application development.

---

## Overview

This project includes multiple components that handle gesture data collection, labeling, model training, and application deployment:

1. **Data Collection**: A script to capture gesture images using a webcam.
2. **Data Labeling**: Automates YOLO-compatible label generation for gesture datasets.
3. **Model Training**: Prepares and trains a YOLOv5 model for gesture recognition.
4. **Application Deployment**: A gesture-controlled application to test the trained model in real-time.

---

## Project Workflow

### 1. Data Collection
- **File**: `data_collection.py`
- **Summary**: Captures webcam images for predefined gestures and saves them in a structured directory.
- **Usage**:
    - Displays live webcam feed with gesture-specific instructions.
    - User presses `r` to start/stop recording frames for a gesture.
    - Move to the next gesture using `n`, and quit using `q`.

### 2. Data Labeling
- **File**: `data_labeling.py`
- **Summary**: Automatically detects hand landmarks and generates YOLO-compatible labels for each gesture image.
- **How It Works**:
    - Utilizes MediaPipe's hand tracking to detect hand bounding boxes.
    - Converts bounding boxes into YOLO format and saves labels alongside images.

### 3. Model Training
- **File**: `training_script.py`
- **Summary**: Prepares the dataset, sets up YOLOv5, and trains a gesture recognition model.
- **Steps**:
    - Splits collected data into training and validation sets.
    - Configures YOLOv5 and installs necessary dependencies.
    - Trains the YOLOv5 model with the preprocessed dataset.

### 4. Gesture-Controlled Application
- **File**: `gestureApp.py`
- **Summary**: Real-time application for controlling an on-screen object based on detected gestures.
- **Key Features**:
    - Uses the trained YOLOv5 model to predict gestures from live webcam feed.
    - Maps gestures to actions (e.g., moving an on-screen object up or down).
    - Visual feedback includes bounding boxes and class labels.

---

## Setup Instructions
Running just the model
1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-repo-url.git
   cd your-repo
   ```
2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Install Dependencies**
   ```bash
   python3 appGesture.py
   ```
Run full pipeline 
1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-repo-url.git
   cd your-repo
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Collect Gesture Data**
   - Run `data_collection.py` to capture images for gestures.

4. **Generate Labels**
   - Run `data_labeling.py` to create YOLO labels.

5. **Train the Model**
   - Run `training_script.py` to prepare and train the YOLOv5 model.

6. **Test the Application**
   - Use `gestureApp.py` to test real-time gesture recognition and control.

---



