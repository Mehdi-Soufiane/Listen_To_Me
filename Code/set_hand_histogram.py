import cv2
import numpy as np
import pickle
from pathlib import Path

class HandCalibrator:
    def __init__(self):
        self.squares_x = 10
        self.squares_y = 5
        self.square_size = 10
        self.square_gap = 10
        
        # Region for squares
        self.start_x = 420
        self.start_y = 140
        
        # Region for hand detection
        self.roi_x = 300
        self.roi_y = 100
        self.roi_w = 300
        self.roi_h = 300
        
        self.camera = None
        self.setup_camera()

    def setup_camera(self):
        """Initialize camera with fallback"""
        self.camera = cv2.VideoCapture(1)
        if not self.camera.isOpened():
            self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            raise ValueError("Could not open any camera")

    def build_calibration_grid(self, img):
        """Create grid of sample squares for skin color calibration"""
        x, y = self.start_x, self.start_y
        imgCrop = None
        crop = None
        
        for i in range(self.squares_y):
            row_crop = None
            for j in range(self.squares_x):
                # Extract square region
                square = img[y:y+self.square_size, x:x+self.square_size]
                
                # Add to row
                if row_crop is None:
                    row_crop = square
                else:
                    row_crop = np.hstack((row_crop, square))
                
                # Draw rectangle on display image
                cv2.rectangle(img, (x,y), 
                            (x+self.square_size, y+self.square_size),
                            (0,255,0), 1)
                
                x += self.square_size + self.square_gap
            
            # Add row to final crop
            if crop is None:
                crop = row_crop
            else:
                crop = np.vstack((crop, row_crop))
            
            # Reset for next row
            x = self.start_x
            y += self.square_size + self.square_gap
            
        return crop

    def calculate_histogram(self, img_crop):
        """Calculate color histogram from sample regions"""
        hsv_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv_crop], [0, 1], None, 
                           [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        return hist

    def process_frame(self, frame, hist=None):
        """Process frame with calculated histogram"""
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (640, 480))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        if hist is not None:
            # Calculate back projection
            dst = cv2.calcBackProject([hsv], [0, 1], hist, 
                                    [0, 180, 0, 256], 1)
            
            # Apply image processing
            disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
            cv2.filter2D(dst, -1, disc, dst)
            
            blur = cv2.GaussianBlur(dst, (11,11), 0)
            blur = cv2.medianBlur(blur, 15)
            
            _, thresh = cv2.threshold(blur, 0, 255,
                                    cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            thresh = cv2.merge((thresh,thresh,thresh))
            
            return frame, thresh
        return frame, None

    def run_calibration(self):
        """Run the calibration process"""
        print("Starting hand histogram calibration...")
        print("Instructions:")
        print("1. Place your hand in the green squares")
        print("2. Press 'c' to capture histogram")
        print("3. Press 's' to save and exit")
        print("4. Press 'q' to quit without saving")
        
        hist = None
        while True:
            ret, frame = self.camera.read()
            if not ret:
                continue
                
            img_crop = self.build_calibration_grid(frame.copy())
            frame, thresh = self.process_frame(frame, hist)
            
            # Show calibration grid
            cv2.imshow("Hand Calibration", frame)
            if thresh is not None:
                cv2.imshow("Threshold", thresh)
            
            key = cv2.waitKey(1)
            
            if key == ord('c'):
                hist = self.calculate_histogram(img_crop)
                print("Histogram captured! Press 's' to save or 'c' to recapture.")
                
            elif key == ord('s') and hist is not None:
                # Save histogram
                with open("hist", "wb") as f:
                    pickle.dump(hist, f)
                print("Histogram saved successfully!")
                break
                
            elif key == ord('q'):
                print("Calibration cancelled.")
                break
        
        self.cleanup()

    def cleanup(self):
        """Release resources"""
        if self.camera is not None:
            self.camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        calibrator = HandCalibrator()
        calibrator.run_calibration()
    except Exception as e:
        print(f"Error during calibration: {str(e)}")