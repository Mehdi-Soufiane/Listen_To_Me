import cv2
from glob import glob
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pickle
import os
from pathlib import Path
from tqdm import tqdm


def load_and_preprocess_dataset(dataset_path='dataset', image_size=(64, 64)):
    """
    Load images and labels from dataset directory and preprocess them.
    """
    images = []
    labels = []
    gesture_map = {}
    
    # Resolve the dataset path relative to the current working directory
    dataset_path = Path(dataset_path).resolve()
    
    if not dataset_path.exists():
        raise ValueError(f"Dataset path '{dataset_path}' does not exist.")
    
    # Get all gesture folders
    gesture_folders = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    
    if not gesture_folders:
        raise ValueError(f"No gesture folders found in {dataset_path}")
    
    print(f"Found {len(gesture_folders)} gesture classes")
    
    # Create gesture to number mapping
    for idx, folder in enumerate(gesture_folders):
        gesture_map[folder.name] = idx
    
    # Save gesture mapping
    with open("gesture_map.pkl", "wb") as f:
        pickle.dump(gesture_map, f)
    print("Saved gesture mapping to gesture_map.pkl")
    
    # Load and preprocess images
    total_images = sum(len(list(folder.glob('*.jpg'))) for folder in gesture_folders)
    print(f"\nLoading and preprocessing {total_images} images...")
    
    with tqdm(total=total_images) as pbar:
        for folder in gesture_folders:
            label = gesture_map[folder.name]
            image_files = list(folder.glob('*.jpg'))
            
            for image_file in image_files:
                img = cv2.imread(str(image_file), 0)  # Read as grayscale
                if img is not None:
                    # Resize image
                    img = cv2.resize(img, image_size)
                    # Normalize pixel values
                    img = img.astype(np.float32) / 255.0
                    images.append(img)
                    labels.append(label)
                    pbar.update(1)
    
    return np.array(images), np.array(labels)


def split_and_save_data(images, labels, output_dir='processed_data'):
    """
    Split dataset into train, validation, and test sets and save them
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Shuffle data
    images, labels = shuffle(images, labels, random_state=42)
    
    # Split data: 70% train, 15% validation, 15% test
    train_images, temp_images, train_labels, temp_labels = train_test_split(
        images, labels, test_size=0.3, random_state=42
    )
    
    val_images, test_images, val_labels, test_labels = train_test_split(
        temp_images, temp_labels, test_size=0.5, random_state=42
    )
    
    # Save datasets
    datasets = {
        'train': (train_images, train_labels),
        'val': (val_images, val_labels),
        'test': (test_images, test_labels)
    }
    
    print("\nSaving datasets:")
    for name, (images, labels) in datasets.items():
        # Save images
        image_file = output_dir / f"{name}_images.pkl"
        with open(image_file, "wb") as f:
            pickle.dump(images, f)
        
        # Save labels
        label_file = output_dir / f"{name}_labels.pkl"
        with open(label_file, "wb") as f:
            pickle.dump(labels, f)
        
        print(f"{name.capitalize()} set - Images: {len(images)}, Labels: {len(labels)}")

def main():
    try:
        print("Starting dataset preparation...")
        
        # Load and preprocess images
        images, labels = load_and_preprocess_dataset()
        
        # Split and save datasets
        split_and_save_data(images, labels)
        
        print("\nDataset preparation completed successfully!")
        print("\nFiles created:")
        print("- gesture_map.pkl (gesture name to number mapping)")
        print("- processed_data/train_images.pkl")
        print("- processed_data/train_labels.pkl")
        print("- processed_data/val_images.pkl")
        print("- processed_data/val_labels.pkl")
        print("- processed_data/test_images.pkl")
        print("- processed_data/test_labels.pkl")
        
    except Exception as e:
        print(f"Error during dataset preparation: {str(e)}")

if __name__ == "__main__":
    main()