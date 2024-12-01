import cv2
import os
import random
import numpy as np
from pathlib import Path

def create_gesture_grid(dataset_path='dataset', grid_columns=5):
    """
    Create a grid visualization of gesture images from the dataset.
    """
    # Resolve dataset path relative to the current working directory
    dataset_path = Path(dataset_path).resolve()
    
    if not dataset_path.exists():
        raise ValueError(f"Dataset path '{dataset_path}' does not exist.")
    
    # Get all gesture folders
    gestures = [d for d in dataset_path.iterdir() if d.is_dir()]
    gestures.sort()  # Sort alphabetically
    
    # Calculate grid dimensions
    n_gestures = len(gestures)
    if n_gestures == 0:
        print("No gesture folders found in dataset")
        return None
        
    rows = (n_gestures + grid_columns - 1) // grid_columns  # Ceiling division
    
    # Get standard image size from first image found
    sample_images = list(gestures[0].glob('*.jpg'))
    if not sample_images:
        print(f"No images found in {gestures[0]}")
        return None
        
    sample_img = cv2.imread(str(sample_images[0]), 0)
    image_x, image_y = sample_img.shape
    
    # Create the visualization grid
    full_img = np.zeros((image_y * rows, image_x * grid_columns), dtype=np.uint8)
    
    print(f"Creating visualization grid with {rows} rows and {grid_columns} columns")
    print(f"Found {n_gestures} gesture classes")
    
    # Fill the grid with gesture images
    for idx, gesture_folder in enumerate(gestures):
        # Calculate position in grid
        row = idx // grid_columns
        col = idx % grid_columns
        
        # Get a random image from the gesture folder
        gesture_images = list(gesture_folder.glob('*.jpg'))
        if gesture_images:
            # Choose random image
            img_path = str(random.choice(gesture_images))
            img = cv2.imread(img_path, 0)
            if img is not None:
                # Resize if needed
                img = cv2.resize(img, (image_x, image_y))
            else:
                img = np.zeros((image_y, image_x), dtype=np.uint8)
        else:
            img = np.zeros((image_y, image_x), dtype=np.uint8)
        
        # Add gesture name to the image
        gesture_name = gesture_folder.name
        cv2.putText(img, gesture_name, (5, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
        
        # Calculate position in the full image
        y_start = row * image_y
        y_end = (row + 1) * image_y
        x_start = col * image_x
        x_end = (col + 1) * image_x
        
        # Place the image in the grid
        full_img[y_start:y_end, x_start:x_end] = img
        
    return full_img

def main():
    try:
        print("Creating gesture dataset visualization...")
        
        # Create visualization
        visualization = create_gesture_grid()
        
        if visualization is not None:
            # Add title to the visualization
            title_height = 60
            title_img = np.ones((title_height, visualization.shape[1]), dtype=np.uint8) * 255
            cv2.putText(title_img, "Gesture Dataset Visualization", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)
            
            # Combine title and visualization
            full_img = np.vstack((title_img, visualization))
            
            # Save the visualization
            cv2.imwrite('gesture_dataset_visualization.jpg', full_img)
            print("Visualization saved as 'gesture_dataset_visualization.jpg'")
            
            # Display the visualization
            cv2.imshow("Gesture Dataset Visualization", full_img)
            print("\nPress any key to close the visualization window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error creating visualization: {str(e)}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
