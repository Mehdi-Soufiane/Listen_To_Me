import numpy as np
import cv2
import os
from glob import glob
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path

# Suppress TensorFlow deprecation warnings
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load_dataset(dataset_path='dataset'):
    """
    Load images from a directory structure where each gesture has its own folder
    Returns images and labels
    """
    images = []
    labels = []
    gesture_map = {}
    
    # Resolve dataset path relative to the current working directory
    dataset_path = Path(dataset_path).resolve()
    
    if not dataset_path.exists():
        raise ValueError(f"Dataset path '{dataset_path}' does not exist.")
    
    gesture_folders = [f for f in dataset_path.iterdir() if f.is_dir()]
    
    print(f"Found {len(gesture_folders)} gesture classes: {[f.name for f in gesture_folders]}")
    
    # Create gesture to number mapping
    for idx, gesture in enumerate(sorted(gesture_folders)):
        gesture_map[gesture.name] = idx
    
    # Save the gesture mapping
    with open('gesture_map.pkl', 'wb') as f:
        pickle.dump(gesture_map, f)
    
    for gesture in gesture_folders:
        gesture_path = gesture
        label = gesture_map[gesture.name]
        
        image_files = list(gesture_path.glob('*.[jp][pn][g]'))
        print(f"Loading {len(image_files)} images for gesture '{gesture.name}'")
        
        for image_file in image_files:
            img = cv2.imread(str(image_file), 0)  # Read as grayscale
            if img is not None:
                img = cv2.resize(img, (64, 64))
                images.append(img)
                labels.append(label)
    
    return np.array(images), np.array(labels), len(gesture_folders)

def prepare_data(images, labels):
    """Prepare data for training"""
    train_images, val_images, train_labels, val_labels = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )
    
    train_images = np.reshape(train_images, (train_images.shape[0], 64, 64, 1)) / 255.0
    val_images = np.reshape(val_images, (val_images.shape[0], 64, 64, 1)) / 255.0
    
    train_labels = to_categorical(train_labels)
    val_labels = to_categorical(val_labels)
    
    return train_images, val_images, train_labels, val_labels

def cnn_model(num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train(dataset_path='dataset'):
    try:
        print("Loading dataset...")
        images, labels, num_classes = load_dataset(dataset_path)
        
        print("\nPreparing data for training...")
        train_images, val_images, train_labels, val_labels = prepare_data(images, labels)
        
        print("\nData shapes:")
        print(f"Training images: {train_images.shape}")
        print(f"Training labels: {train_labels.shape}")
        print(f"Validation images: {val_images.shape}")
        print(f"Validation labels: {val_labels.shape}")
        
        print("\nInitializing model...")
        model = cnn_model(num_classes)
        model.summary()
        
        # Setup callbacks with .keras extension
        callbacks = [
            ModelCheckpoint(
                'best_model.keras',  # Updated extension
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            # Add TensorBoard callback for visualization
            tf.keras.callbacks.TensorBoard(
                log_dir='./logs',
                histogram_freq=1
            )
        ]
        
        print("\nStarting training...")
        history = model.fit(
            train_images,
            train_labels,
            validation_data=(val_images, val_labels),
            epochs=30,
            batch_size=32,
            callbacks=callbacks
        )
        
        # Evaluate the model
        scores = model.evaluate(val_images, val_labels, verbose=0)
        print("\nFinal validation accuracy: %.2f%%" % (scores[1]*100))
        
        # Save the final model with .keras extension
        model.save('final_model.keras')
        print("\nTraining complete. Models saved as 'best_model.keras' and 'final_model.keras'")
        
        # Save training history
        with open('training_history.pkl', 'wb') as f:
            pickle.dump(history.history, f)
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise  # Re-raise the exception for debugging

if __name__ == "__main__":
    print("Starting sign language recognition training...")
    # Add version information for debugging
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras version: {tf.keras.__version__}")
    
    train()
    tf.keras.backend.clear_session()
