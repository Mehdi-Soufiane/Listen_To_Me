import cv2
import numpy as np
import pickle
import os
import sqlite3
import random
from pathlib import Path

image_x, image_y = 64, 64  # Match the dimensions used in training

def get_hand_hist():
    try:
        with open("hist", "rb") as f:
            hist = pickle.load(f)
        return hist
    except FileNotFoundError:
        print("Error: 'hist' file not found. Please run hand histogram calibration first.")
        exit(1)

def init_create_folder_database():
    # Create main dataset directory if it doesn't exist
    if not os.path.exists("dataset"):
        os.mkdir("dataset")
    
    # Create or connect to SQLite database
    conn = sqlite3.connect("gesture_db.db")
    create_table_cmd = """
    CREATE TABLE IF NOT EXISTS gesture (
        g_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
        g_name TEXT NOT NULL UNIQUE
    )"""
    conn.execute(create_table_cmd)
    conn.commit()
    conn.close()

def create_gesture_folder(gesture_name):
    folder_path = Path("dataset") / gesture_name
    if not folder_path.exists():
        folder_path.mkdir(parents=True)
    return folder_path

def store_in_db(g_name):
    conn = sqlite3.connect("gesture_db.db")
    cursor = conn.cursor()
    
    # Check if gesture name already exists
    cursor.execute("SELECT g_id FROM gesture WHERE g_name = ?", (g_name,))
    result = cursor.fetchone()
    
    if result:
        choice = input(f"Gesture '{g_name}' already exists. Want to add more images? (y/n): ")
        if choice.lower() != 'y':
            print("Skipping...")
            conn.close()
            return result[0]
        g_id = result[0]
    else:
        cursor.execute("INSERT INTO gesture (g_name) VALUES (?)", (g_name,))
        g_id = cursor.lastrowid
    
    conn.commit()
    conn.close()
    return g_id

def store_images(gesture_name, g_id):
    total_pics = 1200
    hist = get_hand_hist()
    
    # Try to open camera
    cam = cv2.VideoCapture(1)
    if not cam.isOpened():
        cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Could not open camera")
        return
    
    # ROI coordinates for hand detection
    x, y, w, h = 300, 100, 300, 300

    # Create gesture folder
    folder_path = create_gesture_folder(gesture_name)
    
    # Initialize capture variables
    pic_no = 0
    flag_start_capturing = False
    frames = 0
    
    try:
        while True:
            ret, img = cam.read()
            if not ret:
                continue
                
            img = cv2.flip(img, 1)
            
            # Hand detection and segmentation
            imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
            disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
            cv2.filter2D(dst,-1,disc,dst)
            blur = cv2.GaussianBlur(dst, (11,11), 0)
            blur = cv2.medianBlur(blur, 15)
            thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
            thresh = cv2.merge((thresh,thresh,thresh))
            thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
            thresh = thresh[y:y+h, x:x+w]
            
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            if len(contours) > 0 and flag_start_capturing:
                contour = max(contours, key = cv2.contourArea)
                if cv2.contourArea(contour) > 10000 and frames > 50:
                    x1, y1, w1, h1 = cv2.boundingRect(contour)
                    save_img = thresh[y1:y1+h1, x1:x1+w1]
                    
                    # Make the image square
                    if w1 > h1:
                        save_img = cv2.copyMakeBorder(
                            save_img, 
                            int((w1-h1)/2), 
                            int((w1-h1)/2), 
                            0, 0, 
                            cv2.BORDER_CONSTANT, 
                            (0, 0, 0)
                        )
                    elif h1 > w1:
                        save_img = cv2.copyMakeBorder(
                            save_img, 
                            0, 0, 
                            int((h1-w1)/2), 
                            int((h1-w1)/2), 
                            cv2.BORDER_CONSTANT, 
                            (0, 0, 0)
                        )
                    
                    # Resize to standard size
                    save_img = cv2.resize(save_img, (image_x, image_y))
                    
                    # Random horizontal flip for data augmentation
                    if random.random() > 0.5:
                        save_img = cv2.flip(save_img, 1)
                    
                    pic_no += 1
                    
                    # Save image
                    cv2.imwrite(str(folder_path / f"{pic_no}.jpg"), save_img)
                    
                    # Show capture status
                    cv2.putText(
                        img, 
                        f"Capturing... {pic_no}/{total_pics}", 
                        (30, 60), 
                        cv2.FONT_HERSHEY_TRIPLEX, 
                        1, 
                        (127, 255, 255),
                        2
                    )

            # Draw ROI rectangle
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
            
            # Show progress
            status_text = "Press 'c' to start/pause capture" if not flag_start_capturing else f"Capturing: {pic_no}/{total_pics}"
            cv2.putText(img, status_text, (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1, (127, 127, 255), 2)
            
            # Display windows
            cv2.imshow("Capturing gesture", img)
            cv2.imshow("Threshold", thresh)
            
            # Handle key presses
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('c'):
                flag_start_capturing = not flag_start_capturing
                frames = 0
            
            if flag_start_capturing:
                frames += 1
            
            if pic_no >= total_pics:
                break
                
    finally:
        cam.release()
        cv2.destroyAllWindows()

def main():
    print("Gesture Data Collection Tool")
    print("-" * 30)
    
    init_create_folder_database()
    
    while True:
        gesture_name = input("Enter gesture name (or 'q' to quit): ").strip()
        if gesture_name.lower() == 'q':
            break
            
        if not gesture_name:
            print("Error: Gesture name cannot be empty")
            continue
            
        g_id = store_in_db(gesture_name)
        print(f"Collecting images for gesture: {gesture_name} (ID: {g_id})")
        print("Press 'c' to start/pause capture, 'q' to finish")
        store_images(gesture_name, g_id)
        
        print(f"Completed collecting images for {gesture_name}")
        print("-" * 30)

if __name__ == "__main__":
    main()