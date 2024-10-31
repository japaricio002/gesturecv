# train.py
import subprocess
import sys
import os
from pathlib import Path
import torch
import shutil
import random
import yaml

def count_images(dataset_path):
    """Count total number of images in dataset"""
    image_extensions = ('.jpg', '.jpeg', '.png')
    total_images = 0
    
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(image_extensions):
                total_images += 1
    
    return total_images

def check_m1_device():
    """Check and configure M1 device settings"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using M1 GPU acceleration via Metal Performance Shaders (MPS)")
    else:
        device = torch.device("cpu")
        print("MPS not available, using CPU")
    return device

def setup_environment():
    """Setup training environment and dependencies"""
    data_yaml = {
        'path': 'dataset',
        'train': 'images/train',
        'val': 'images/val',
        'nc': 5,
        'names': ['thumbs_up', 'thumbs_down', 'peace', 'stop', 'okay']
    }
    
    with open('data.yaml', 'w') as f:
        yaml.dump(data_yaml, f)

def split_dataset(dataset_path, train_ratio=0.8):
    """Split dataset into train and validation sets"""
    # Create directory structure
    images_train = Path('dataset/images/train')
    images_val = Path('dataset/images/val')
    labels_train = Path('dataset/labels/train')
    labels_val = Path('dataset/labels/val')
    
    for path in [images_train, images_val, labels_train, labels_val]:
        path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = []
    for gesture in os.listdir(dataset_path):
        gesture_path = Path(dataset_path) / gesture
        if gesture_path.is_dir():
            image_files.extend(list(gesture_path.glob('*.jpg')))
    
    if not image_files:
        print("No images found in the dataset directory!")
        sys.exit(1)
    
    # Shuffle and split
    random.shuffle(image_files)
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    print(f"Splitting dataset: {len(train_files)} training images, {len(val_files)} validation images")
    
    # Copy files to new structure
    for files, img_path, label_path in [(train_files, images_train, labels_train),
                                      (val_files, images_val, labels_val)]:
        for img_file in files:
            # Copy image
            shutil.copy(str(img_file), str(img_path / img_file.name))
            
            # Copy corresponding label
            label_file = img_file.with_suffix('.txt')
            if label_file.exists():
                shutil.copy(str(label_file), str(label_path / label_file.name))

def setup_yolov5():
    """Clone and setup YOLOv5"""
    if not Path('yolov5').exists():
        print("Cloning YOLOv5 repository...")
        try:
            subprocess.check_call(['git', 'clone', 'https://github.com/ultralytics/yolov5.git'])
        except subprocess.CalledProcessError as e:
            print(f"Error cloning YOLOv5: {e}")
            sys.exit(1)
    
    print("Installing YOLOv5 requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "yolov5/requirements.txt"])
    except subprocess.CalledProcessError as e:
        print(f"Error installing YOLOv5 requirements: {e}")
        sys.exit(1)

def train_model():
    """Train YOLOv5 model with M1-optimized settings"""
    print("Starting model training...")
    
    # Optimized batch size for M1
    batch_size = 8
    
    try:
        subprocess.check_call([
            sys.executable, 'yolov5/train.py',
            '--img', '640',
            '--batch', str(batch_size),
            '--epochs', '100',
            '--data', 'data.yaml',
            '--weights', 'yolov5s.pt',
            '--project', 'runs/train',
            '--name', 'gesture_detection',
            '--cache'
        ])
    except subprocess.CalledProcessError as e:
        print(f"Error during training: {e}")
        sys.exit(1)

def estimate_training_time(num_images):
    """Estimate training time based on dataset size"""
    images_per_second = 15  # Conservative estimate for M1
    epochs = 100
    
    total_images = num_images * epochs
    seconds = total_images / images_per_second
    hours = seconds / 3600
    
    return hours

def main():
    if not Path('dataset').exists():
        print("Error: 'dataset' directory not found!")
        print("Please run the data collection script first.")
        sys.exit(1)
    
    # Count images using the new function
    num_images = count_images('dataset')
    print(f"\nFound {num_images} images in dataset")
    
    if num_images == 0:
        print("No images found in the dataset directory!")
        sys.exit(1)
    
    # Estimate training time
    estimated_hours = estimate_training_time(num_images)
    print(f"Estimated training time on M1 MacBook Pro: {estimated_hours:.1f} hours")
    print("Note: This is a rough estimate and actual time may vary\n")
    
    proceed = input("Would you like to proceed with training? (y/n): ")
    if proceed.lower() != 'y':
        print("Training cancelled")
        sys.exit(0)
    
    try:
        # Step 1: Check M1 device
        device = check_m1_device()
        
        # Step 2: Setup environment and create yaml file
        setup_environment()
        
        # Step 3: Split dataset
        split_dataset('dataset')
        
        # Step 4: Setup YOLOv5
        setup_yolov5()
        
        # Step 5: Train model
        train_model()
        
        print("\nTraining completed successfully!")
        print("Your trained model can be found in: runs/train/gesture_detection/")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()