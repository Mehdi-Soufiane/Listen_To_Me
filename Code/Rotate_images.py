import cv2
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np


def augment_images(dataset_path='dataset'):
    """
    Augment images in the dataset with various transformations.
    Applies the following augmentations only to non-augmented images:
    - Horizontal flip
    - Random rotation
    - Adding Gaussian noise
    - Brightness variation
    """
    augmentation_types = ['flip', 'rotate', 'noise', 'brightness']
    dataset_path = Path(dataset_path).resolve()  # Resolve to absolute path for consistency
    
    if not dataset_path.exists():
        raise ValueError(f"Dataset path '{dataset_path}' does not exist.")
    
    gesture_folders = [d for d in dataset_path.iterdir() if d.is_dir()]
    
    print(f"Found {len(gesture_folders)} gesture classes")
    
    # Count total original images to process
    total_images = sum(len(list(folder.glob('*.jpg'))) for folder in gesture_folders)
    print(f"Processing {total_images} images...")

    for gesture_folder in gesture_folders:
        print(f"\nProcessing images in {gesture_folder.name}")
        
        # Only process non-augmented images (exclude already augmented ones)
        image_files = [
            img_path for img_path in gesture_folder.glob('*.jpg')
            if not any(suffix in img_path.stem for suffix in ['_flipped', '_rotated', '_noisy', '_bright'])
        ]
        
        with tqdm(total=len(image_files) * len(augmentation_types)) as pbar:
            for img_path in image_files:
                img = cv2.imread(str(img_path), 0)
                if img is None:
                    print(f"Warning: Could not read {img_path}")
                    continue
                
                base_filename = img_path.stem
                
                # Horizontal flip
                flipped = cv2.flip(img, 1)
                new_path = gesture_folder / f"{base_filename}_flipped.jpg"
                if not new_path.exists():  # Avoid duplicates
                    cv2.imwrite(str(new_path), flipped)
                pbar.update(1)
                
                # Random rotation between -15 and 15 degrees
                angle = np.random.uniform(-15, 15)
                height, width = img.shape
                center = (width / 2, height / 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(img, rotation_matrix, (width, height))
                new_path = gesture_folder / f"{base_filename}_rotated.jpg"
                if not new_path.exists():  # Avoid duplicates
                    cv2.imwrite(str(new_path), rotated)
                pbar.update(1)
                
                # Add Gaussian noise
                noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
                noisy = cv2.add(img, noise)
                new_path = gesture_folder / f"{base_filename}_noisy.jpg"
                if not new_path.exists():  # Avoid duplicates
                    cv2.imwrite(str(new_path), noisy)
                pbar.update(1)
                
                # Random brightness adjustment
                brightness = np.random.uniform(0.7, 1.3)
                adjusted = cv2.multiply(img, brightness)
                adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
                new_path = gesture_folder / f"{base_filename}_bright.jpg"
                if not new_path.exists():  # Avoid duplicates
                    cv2.imwrite(str(new_path), adjusted)
                pbar.update(1)

def main():
    try:
        print("Starting image augmentation...")
        
        # Run augmentation (relative dataset path)
        augment_images(dataset_path='dataset')  # 'dataset' folder in the current directory
        
        print("\nAugmentation completed successfully!")
        
    except Exception as e:
        print(f"Error during augmentation: {str(e)}")

def main():
    try:
        print("Starting image augmentation...")
        
        # Run augmentation
        augment_images()
        
        print("\nAugmentation completed successfully!")
        
    except Exception as e:
        print(f"Error during augmentation: {str(e)}")

if __name__ == "__main__":
    main()
