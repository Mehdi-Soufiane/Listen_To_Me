import sys
import cv2
import numpy as np
import pickle
import os
import sqlite3
import subprocess
import pyttsx3
from pathlib import Path
from threading import Thread
from tensorflow.keras.models import load_model
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QMessageBox, QStackedWidget, QPushButton, QInputDialog, QProgressDialog
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QEvent, QThread
from PyQt6.QtGui import QImage, QPixmap, QShortcut, QKeySequence

class CalibrationWidget(QWidget):
    finished = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.squares_x = 5
        self.squares_y = 5
        self.square_size = 30
        self.square_gap = 20
        self.start_x = 320
        self.start_y = 100
        self.camera = None
        self.hist = None
        self.setup_ui()

        # Ensure widget can receive focus
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFocus()

        # Install an event filter for global key event handling
        self.installEventFilter(self)

        # Alternative key handling with QShortcut
        self.setup_shortcuts()

    def setup_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Instructions
        instructions = QLabel("Press 'A' to capture histogram\nPress 'S' to save\nPress 'Q' to quit")
        instructions.setStyleSheet("font-size: 14px; color: #333;")
        instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(instructions)
        
        # Camera feed
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)
        layout.addWidget(self.video_label)

    def setup_shortcuts(self):
        QShortcut(QKeySequence('A'), self).activated.connect(self.capture_histogram)
        QShortcut(QKeySequence('S'), self).activated.connect(self.save_histogram)
        QShortcut(QKeySequence('Q'), self).activated.connect(lambda: [self.cleanup(), self.finished.emit()])

    def start_camera(self):
        self.camera = cv2.VideoCapture(1)
        if not self.camera.isOpened():
            self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            QMessageBox.critical(self, "Error", "Could not open camera")
            return
            
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)
        
    def keyPressEvent(self, event):
        print("Key Press Detected:", event.key())  # Debugging key presses
        if event.key() == Qt.Key.Key_A:
            print("A pressed")
            self.capture_histogram()
        elif event.key() == Qt.Key.Key_S:
            print("S pressed")
            self.save_histogram()
        elif event.key() == Qt.Key.Key_Q:
            print("Q pressed")
            self.cleanup()
            self.finished.emit()

    def eventFilter(self, source, event):
        if event.type() == QEvent.Type.KeyPress:
            print(f"EventFilter Key Pressed: {event.key()}")
            if event.key() == Qt.Key.Key_A:
                self.capture_histogram()
                return True
            elif event.key() == Qt.Key.Key_S:
                self.save_histogram()
                return True
            elif event.key() == Qt.Key.Key_Q:
                self.cleanup()
                self.finished.emit()
                return True
        return super().eventFilter(source, event)

    def update_frame(self):
        ret, frame = self.camera.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (640, 480))
            
            # Draw calibration grid
            x, y = self.start_x, self.start_y
            for i in range(self.squares_y):
                for j in range(self.squares_x):
                    cv2.rectangle(frame, 
                                  (x, y), 
                                  (x + self.square_size, y + self.square_size),
                                  (0, 255, 0), 1)
                    x += self.square_size + self.square_gap
                x = self.start_x
                y += self.square_size + self.square_gap
            
            # Show preview of skin detection if histogram exists
            if self.hist is not None:
                processed = self.process_frame(frame.copy())
                frame = np.hstack((frame, processed))
            
            # Convert to Qt format
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_image))
            
    def capture_histogram(self):
        if self.camera is None:
            return
            
        ret, frame = self.camera.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (640, 480))
            
            # Create grid and capture histogram
            x, y = self.start_x, self.start_y
            hsvCrop = None
            
            for i in range(self.squares_y):
                for j in range(self.squares_x):
                    roi = frame[y:y+self.square_size, x:x+self.square_size]
                    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    if hsvCrop is None:
                        hsvCrop = hsv_roi
                    else:
                        hsvCrop = np.append(hsvCrop, hsv_roi, axis=0)
                    x += self.square_size + self.square_gap
                x = self.start_x
                y += self.square_size + self.square_gap
            
            # Calculate histogram
            self.hist = cv2.calcHist([hsvCrop], [0, 1], None, [180, 256], [0, 180, 0, 256])
            cv2.normalize(self.hist, self.hist, 0, 255, cv2.NORM_MINMAX)
            QMessageBox.information(self, "Success", "Histogram captured! Check the preview to see if it works well.")
            
    def process_frame(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0, 1], self.hist, [0, 180, 0, 256], 1)
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        cv2.filter2D(dst, -1, disc, dst)
        blur = cv2.GaussianBlur(dst, (11, 11), 0)
        blur = cv2.medianBlur(blur, 15)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        
    def save_histogram(self):
        if self.hist is not None:
            with open("hist", "wb") as f:
                pickle.dump(self.hist, f)
            QMessageBox.information(self, "Success", "Histogram saved successfully!")
            self.cleanup()
            self.finished.emit()
        else:
            QMessageBox.warning(self, "Warning", "Please capture histogram first!")
            
    def cleanup(self):
        if self.camera is not None:
            self.timer.stop()
            self.camera.release()

class GestureRecorder(QWidget):
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.camera = None
        self.recording = False
        self.total_pics = 1200
        self.pic_no = 0
        self.gesture_name = ""
        self.gesture_id = None
        self.training_worker = None
        self.progress_dialog = None
        self.setup_ui()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setup_shortcuts()

        # Initialize the SQLite database and dataset folder
        self.init_create_folder_database()

    def init_create_folder_database(self):
        """Initialize SQLite database and dataset directory."""
        # Create dataset directory
        if not os.path.exists("dataset"):
            os.mkdir("dataset")
        
        # Initialize SQLite database
        conn = sqlite3.connect("gesture_db.db")
        create_table_cmd = """
        CREATE TABLE IF NOT EXISTS gesture (
            g_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
            g_name TEXT NOT NULL UNIQUE
        )"""
        conn.execute(create_table_cmd)
        conn.commit()
        conn.close()

    def store_in_db(self, gesture_name):
        """Store the gesture name in the database, if not already present."""
        conn = sqlite3.connect("gesture_db.db")
        cursor = conn.cursor()

        # Check if the gesture already exists
        cursor.execute("SELECT g_id FROM gesture WHERE g_name = ?", (gesture_name,))
        result = cursor.fetchone()

        if result:
            choice = QMessageBox.question(
                self, 
                "Gesture Exists",
                f"Gesture '{gesture_name}' already exists. Add more images to it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if choice == QMessageBox.StandardButton.Yes:
                g_id = result[0]
            else:
                conn.close()
                return None
        else:
            cursor.execute("INSERT INTO gesture (g_name) VALUES (?)", (gesture_name,))
            g_id = cursor.lastrowid

        conn.commit()
        conn.close()
        return g_id

    def setup_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Instructions
        instructions = QLabel("Press 'C' to start/stop recording\nPress 'Q' to quit")
        instructions.setStyleSheet("font-size: 14px; color: #333;")
        instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(instructions)

        # Camera feed
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)
        layout.addWidget(self.video_label)

    def setup_shortcuts(self):
        QShortcut(QKeySequence('C'), self).activated.connect(self.toggle_recording)
        QShortcut(QKeySequence('Q'), self).activated.connect(self.finish_recording)

    def start_camera(self, gesture_name):
        self.gesture_name = gesture_name
        self.gesture_id = self.store_in_db(gesture_name)

        if self.gesture_id is None:
            self.finished.emit()  # Return to main menu if user opts out
            return

        self.camera = cv2.VideoCapture(1)
        if not self.camera.isOpened():
            self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            QMessageBox.critical(self, "Error", "Could not open camera")
            return

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.camera.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (640, 480))

            # Draw ROI rectangle
            cv2.rectangle(frame, (300, 100), (600, 400), (0, 255, 0), 2)

            if self.recording:
                cv2.putText(frame, f"Recording: {self.pic_no}/{self.total_pics}",
                            (30, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 2)

            # Convert to Qt format
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_image))

            if self.recording:
                self.save_image(frame)

    def toggle_recording(self):
        self.recording = not self.recording
        if self.recording:
            # Create folder for the gesture
            folder_path = Path("dataset") / self.gesture_name
            folder_path.mkdir(parents=True, exist_ok=True)

    def save_image(self, frame):
        if self.pic_no < self.total_pics:
            roi = frame[100:400, 300:600]

            # Save the image in the corresponding gesture folder
            folder_path = Path("dataset") / self.gesture_name
            save_path = folder_path / f"{self.pic_no}.jpg"
            cv2.imwrite(str(save_path), roi)
            self.pic_no += 1

            if self.pic_no >= self.total_pics:
                self.recording = False
                self.finish_recording()
        else:
            self.recording = False

    def cleanup(self):
        if self.camera is not None:
            self.timer.stop()
            self.camera.release()

    def finish_recording(self):
        self.cleanup()

        if self.pic_no > 0:
            response = QMessageBox.question(
                self,
                "Training",
                "Would you like to retrain the model now with the new gestures?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if response == QMessageBox.StandardButton.Yes:
                self.start_training_process()

        self.finished.emit()

    def start_training_process(self):
        self.training_worker = TrainingWorker()
        self.training_worker.progress.connect(self.update_training_progress)
        self.training_worker.finished.connect(self.on_training_finished)

        # Progress dialog
        self.progress_dialog = QProgressDialog("Starting training...", "Cancel", 0, 0, self)
        self.progress_dialog.setWindowTitle("Training Progress")
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.show()

        # Start training worker thread
        self.training_worker.start()

    def update_training_progress(self, message):
        self.progress_dialog.setLabelText(message)

    def on_training_finished(self, success):
        self.progress_dialog.close()
        if success:
            QMessageBox.information(self, "Success", "Training completed successfully!")
        else:
            QMessageBox.critical(self, "Error", "Training failed. Please check the logs.")


class RecognitionWidget(QWidget):
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        instructions = QLabel("Click the button below to start recognition.")
        instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(instructions)

        self.start_button = QPushButton("Start Recognition")
        self.start_button.clicked.connect(self.run_recognition)
        layout.addWidget(self.start_button)

    def run_recognition(self):
        try:
            # Run the final.py script
            subprocess.run(['python', 'final.py'], check=True)
        except subprocess.CalledProcessError as e:
            QMessageBox.critical(self, "Error", f"Recognition failed: {str(e)}")

class TrainingWorker(QThread):
    progress = pyqtSignal(str)  # Signal to update progress messages
    finished = pyqtSignal(bool)  # Signal to indicate completion

    def run(self):
        scripts = ['Rotate_images.py', 'load_images.py', 'create_gestures.py', 'cnn_model_train.py']
        try:
            for script in scripts:
                self.progress.emit(f"Running {script}...")

                # Run the script with real-time output capture
                process = subprocess.Popen(
                    ['python', script],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                # Capture output line by line
                for line in process.stdout:
                    self.progress.emit(line.strip())

                # Check for errors
                process.wait()
                if process.returncode != 0:
                    for line in process.stderr:
                        self.progress.emit(line.strip())
                    raise Exception(f"Error in {script}")
                
            self.finished.emit(True)  # Training successful
        except Exception as e:
            self.progress.emit(f"Error: {str(e)}")
            self.finished.emit(False)  # Training failed

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sign Language Recognition System")
        self.setup_ui()

    def setup_ui(self):
        # Central widget and layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Header
        header = QLabel("Sign Language Recognition System")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet("font-size: 24px; font-weight: bold; margin: 20px;")
        layout.addWidget(header)

        # Navigation buttons
        button_layout = QHBoxLayout()
        self.calibrate_btn = QPushButton("Calibrate Skin Tone")
        self.record_btn = QPushButton("Record New Gesture")
        self.recognize_btn = QPushButton("Start Recognition")
        self.display_btn = QPushButton("Display Gestures")

        for btn in [self.calibrate_btn, self.record_btn, self.recognize_btn, self.display_btn]:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    padding: 10px 20px;
                    border: none;
                    border-radius: 4px;
                    font-size: 14px;
                    min-width: 150px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
            """)
            button_layout.addWidget(btn)

        layout.addLayout(button_layout)

        # Stacked widget for different functionalities
        self.stacked_widget = QStackedWidget()
        self.calibration_widget = CalibrationWidget()
        self.gesture_recorder = GestureRecorder()
        self.recognition_widget = RecognitionWidget()

        self.stacked_widget.addWidget(self.create_welcome_screen())
        self.stacked_widget.addWidget(self.calibration_widget)
        self.stacked_widget.addWidget(self.gesture_recorder)
        self.stacked_widget.addWidget(self.recognition_widget)

        layout.addWidget(self.stacked_widget)

        # Connect buttons to actions
        self.calibrate_btn.clicked.connect(self.show_calibration)
        self.record_btn.clicked.connect(self.show_gesture_recorder)
        self.recognize_btn.clicked.connect(self.show_recognition)
        self.display_btn.clicked.connect(self.run_display_gestures)

        # Connect "finished" signals
        self.calibration_widget.finished.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        self.gesture_recorder.finished.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        self.recognition_widget.finished.connect(lambda: self.stacked_widget.setCurrentIndex(0))

    def create_welcome_screen(self):
        welcome = QWidget()
        layout = QVBoxLayout(welcome)
        message = QLabel("""
            Welcome to the Sign Language Recognition System!

            Get started by:
            1. Calibrating your skin tone
            2. Recording gesture samples
            3. Starting recognition
            4. Displaying recorded gestures
        """)
        message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(message)
        return welcome

    def show_calibration(self):
        self.stacked_widget.setCurrentIndex(1)
        self.calibration_widget.start_camera()

    def show_gesture_recorder(self):
        text, ok = QInputDialog.getText(self, 'New Gesture', 'Enter gesture name:')
        if ok and text:
            self.stacked_widget.setCurrentIndex(2)
            self.gesture_recorder.start_camera(text)

    def show_recognition(self):
        self.stacked_widget.setCurrentIndex(3)
        self.recognition_widget.run_recognition()

    def run_display_gestures(self):
        try:
            subprocess.run(['python', 'display_gestures.py'], check=True)
        except subprocess.CalledProcessError as e:
            QMessageBox.critical(self, "Error", f"Displaying gestures failed: {str(e)}")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1300, 800)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
