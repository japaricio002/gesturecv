# Gesture Recognition Project

A comprehensive system for collecting, labeling, and training a gesture recognition model using OpenCV, MediaPipe, and YOLOv5.

## 📝 Overview

This project consists of three main components:
1. Data Collection (`data_collection.py`)
2. Data Labeling (`data_labeling.py`)
3. Model Training (`training_script.py`)


## 📚 Components

### 1. Data Collection (`data_collection.py`)

A tool designed to automate the collection of image data for gesture recognition using OpenCV, with each function serving a purpose in managing the camera, organizing data storage, and controlling recording interactions.    

#### Features:
- Automated image capture and organization
- Real-time display of recording status
- Support for multiple gesture classes
- Cross-platform camera compatibility (Windows, macOS, Linux)

#### Controls:
- `R`: Start/Stop recording
- `N`: Move to next gesture
- `Q`: Quit collection

### \__init__(self, base_path="dataset")
The constructor (\__init__) initializes an instance of DataCollector with the following properties:

**self.base_path**: The directory where data (images) will be saved. Default is "dataset".
**self.gestures**: A list of gestures to be recorded, e.g., "thumbs_up", "peace".
**self.current_gesture**, **self.frame_count**, and **self.recording**: Variables to track the current gesture being recorded, the number of frames collected for that gesture, and the recording state (whether or not frames are actively being saved).
**self.setup_camera()**: Initializes the camera by calling setup_camera as soon as an instance is created.

### setup_camera(self)
This method sets up the camera using **cv2.VideoCapture()**:
* Detects the operating system (**platform.system()**), configuring the camera differently for macOS (Darwin) and Windows/Linux.
* macOS uses **cv2.CAP_AVFOUNDATION** as its backend, while Windows/Linux uses the default.
* Waits 2 seconds to allow the camera to initialize (**time.sleep(2)**).
* Checks if the camera is open and able to capture frames. If not, it displays troubleshooting tips and exits the program.
* Attempts to read a test frame to ensure the camera works, releasing it if this fails.
Purpose: Ensures the camera is ready and working before proceeding to capture images.

### create_directories(self)
This method creates a folder for each gesture in self.gestures inside self.base_path:
* Iterates through each gesture in self.gestures, creating a directory (if it doesn’t already exist) for that gesture within the specified base path.
* Prints confirmation of each directory created.
Purpose: Organizes gesture data by creating a folder structure to save images for each gesture.

### collect_data(self, frames_per_gesture=100)
This is the main function for collecting data through the webcam:
1. Initial Setup:
    * Displays controls and initializes directories by calling **create_directories()**.
2. Loop for Each Gesture:
    * While there are gestures left to record, it captures frames from the webcam.
3. Overlay Display:
    * Uses **cv2.putText()** to display information on the screen, such as the current gesture, frame progress, recording status, and controls (R to record/pause, N for next gesture, Q to quit).
4. Key Input Handling:
    * r: Toggles self.recording. When recording is True, frames are saved.
    * n: Moves to the next gesture, resets the frame count, and pauses recording.
    * q: Exits the loop.
5. Recording Frames:
    * If self.recording is True, the current frame is saved to a file. Each file is named with the gesture, a timestamp, and saved in the gesture’s folder.
    * Increments **self.frame_count** each time a frame is saved. When the count reaches **frames_per_gesture**, it automatically moves to the next gesture.
6. Exit and Cleanup:
    * Releases the camera (**self.cap.release()**) and closes any OpenCV windows (**cv2.destroyAllWindows()**).
    * Prints the location of saved data.
Purpose: Collects labeled image data for each gesture by saving frames when recording is active.

### 2. Data Labeling (`data_labeling.py`)

Processes collected images using MediaPipe Hands to generate YOLO format labels.

#### Features:
- Hand landmark detection
- Automatic bounding box calculation
- YOLO format label generation
- Batch processing support

## data_labeling.py
### \__init__(self)
The constructor (\__init__) initializes the MediaPipe Hands model:

**self.mp_hands** = **mp.solutions.hands**: Initializes access to the MediaPipe Hands module.
**self.hands** = **self.mp_hands.Hands(...)**: Configures the Hands model with:
**static_image_mode=True**: Enables static image processing.
**max_num_hands=1**: Limits detection to a single hand.
**min_detection_confidence=0.5**: Requires at least 50% confidence to consider a detection valid.
**Purpose**: Initializes the MediaPipe Hands model for hand landmark detection in static images.

### detect_hand(self, image)
This method detects hand landmarks and calculates a bounding box around the detected hand:

Convert Image to RGB:
**image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)**: Converts the image from BGR (OpenCV's default) to RGB (MediaPipe's required format).
Process Image with MediaPipe:
**results = self.hands.process(image_rgb)**: Runs the hand detection model on the image.
Bounding Box Calculation:
If hand landmarks are detected (if **results.multi_hand_landmarks**):
Retrieves the height and width of the image and the detected landmarks.
Extracts x and y coordinates for each landmark and calculates minimum and maximum x and y values to form a bounding box.
Adds padding of 20 pixels to each side of the bounding box for buffer space.
Returns the bounding box coordinates [**x_min**, **y_min**, **x_max**, **y_max**].
If no landmarks are found, returns **None**.
**Purpos**e: Detects and returns the bounding box of a hand in the image, or None if no hand is detected.

### create_yolo_label(self, bbox, image_shape, class_id)
This function converts the bounding box into YOLO format, which uses relative coordinates:

Calculate Relative Center and Size:
Extracts height (h) and width (w) of the image from image_shape.
Calculates the center of the bounding box in the image by averaging **x_min** and **x_max** for **x_center** and **y_min** and **y_max for y_center**.
Calculates the bounding box width and height as percentages of the image’s dimensions.
Format YOLO Label:
Returns a string in YOLO format: **class_id** **x_center** **y_center** width height.
**Purpose**: Converts absolute bounding box coordinates into YOLO format for training object detection models.

### process_dataset(self, dataset_path)
Processes all images in the dataset, detects hands, and creates YOLO labels:

Iterate Over Gestures:
Gets a list of gesture names (**gestures = os.listdir(dataset_path)**) and iterates over them, treating each folder as a unique gesture class.
**gesture_id** is assigned as the class ID (index in gestures list).
Process Each Image in Gesture Folder:
Iterates through images (**image_file**) in each gesture folder.
Checks that the file is an image by verifying the extension (.jpg, .jpeg, .png).
Hand Detection and Label Creation:
Reads each image with **cv2.imread()**.
Calls **detect_hand()** to detect the bounding box.
If a hand is detected (bbox exists), it:
Converts the bounding box to YOLO format by calling **create_yolo_label()** with **gesture_id**.
Saves the label in a .txt file with the same name as the image, in the same directory.
If no hand is detected, it prints a message indicating the issue.
Purpose: Processes an entire dataset of gesture images, creating YOLO labels for images with detected hands to aid in training an object detection model.

### \__main__ Block
The \__main__ block is executed if the script runs directly:

Instantiates a GestureLabeler object and calls **process_dataset()** on the "dataset" directory, starting the label creation process.
**Purpose**: Serves as an entry point to run the script, processing gesture data for YOLO training.


### 3. Training Script (`training_script.py`)

Handles the complete training pipeline using YOLOv5.

#### Features:
- Automatic environment setup
- Dataset splitting (train/validation)
- M1 GPU optimization
- Training time estimation
- YOLOv5 integration

## ⚙️ Configuration

The training configuration is automatically generated in `data.yaml`:
- Training/Validation split: 80/20
- Image size: Defined in training script
- Batch size: Optimized for your hardware
- Number of epochs: Configurable in training script

## 🔄 Training Process

1. **Preparation**:
   - Set up YOLOv5
   - Split dataset
   - Generate configuration

2. **Training**:
   - Automatically selects appropriate device (GPU/CPU)
   - Estimates training time
   - Provides progress updates
   - Saves model checkpoints

## 🚀 Performance Optimization

- M1 GPU support through Metal Performance Shaders (MPS)
- Automatic hardware detection and configuration
- Optimized batch size selection

## training_script.py

### count_images(dataset_path)
Counts the total number of images in the given dataset directory.

Define Valid Extensions:
image_extensions = ('.jpg', '.jpeg', '.png'): Accepts common image formats.
Walk Through Directory:
Uses os.walk to traverse subdirectories and count files with valid image extensions.
Return Count:
Returns total_images, the total count of images in the directory.
Purpose: Ensures the dataset has a sufficient number of images for training.

### check_m1_device()
Checks if the script is running on an Apple M1 device and configures it accordingly.

MPS Availability Check:
Checks if Metal Performance Shaders (MPS) are available for GPU acceleration on M1.
Return Device:
If MPS is available, it returns an MPS device for GPU usage; otherwise, defaults to CPU.
Purpose: Detects M1-specific hardware to optimize training.

### setup_environment()
Creates the **data.yaml** file needed for YOLO training.

Define Dataset Info:
Configures paths to the dataset, the number of classes (nc), and class names (names).
Write YAML File:
Saves this information in data.yaml for YOLOv5 training.
Purpose: Sets up the environment by creating a configuration file to specify dataset paths and labels.

### split_dataset(dataset_path, train_ratio=0.8)
Splits the dataset into training and validation sets.

Create Directory Structure:
Creates folders for train and validation images and labels under dataset/images and dataset/labels.
Collect Image Files:
Scans **dataset_path** to gather all image files by gesture class.
Shuffle and Split Files:
Shuffles the images and splits them based on **train_ratio**.
Copy Files:
Copies images and their corresponding label files to either the train or validation directories.
Purpose: Organizes data into separate training and validation sets for YOLOv5.

### setup_yolov5()
Sets up YOLOv5 by cloning its repository and installing dependencies.

Clone YOLOv5:
If YOLOv5 isn’t already cloned, it downloads it from GitHub.
Install YOLOv5 Requirements:
Installs required libraries specified in yolov5/requirements.txt.
Purpose: Ensures YOLOv5 is set up and ready for training.

### train_model()
Starts the YOLOv5 model training process.

Set Parameters:
Specifies the batch size, image size, number of epochs, and path to the data file.
Start Training:
Calls the YOLOv5 training script (**yolov5/train.py**) with the specified parameters.
Error Handling:
If training fails, it prints an error message and exits.
Purpose: Executes the training of the YOLOv5 model on the prepared dataset.

### estimate_training_time(num_images)
Estimates the total training time based on the dataset size.

Calculate Total Images:
Multiplies num_images by epochs to get the total images processed.
Estimate Time in Hours:
Divides by an estimated processing speed and converts to hours.
Purpose: Provides a rough estimate of the training duration.

### main()
Coordinates the execution of all functions in a logical order:

Check Dataset:
Confirms the dataset directory exists and counts images.
Estimate Training Time:
Calculates and displays the estimated training time.
Prompt for Confirmation:
Asks the user if they wish to proceed with training.
Run Setup and Training Steps:
Executes each step, from installing requirements to training the model.
Error Handling:
If any error occurs during execution, it displays the error and exits.