# Listen To Me - Sign Language Recognition System

> Fast communication, Easy life - Breaking barriers in deaf communication through AI

[![Python Version](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## About The Project

Listen To Me is an innovative sign language recognition system developed during the MoroccoAI InnovAI Hackathon 2024. The project aims to bridge the communication gap for over 466 million people worldwide with disabling hearing loss, including 300,000 individuals in Morocco.

### The Problem
- Communication barrier for deaf individuals worldwide
- Limited access to real-time translation between sign language and spoken language
- Lack of seamless integration between deaf and hearing communities

### Our Solution
Our system leverages computer vision and deep learning to provide:
- Real-time sign language recognition 
- Voice-to-text conversion
- Text-to-speech functionality
- Calculator mode through gesture recognition
- Custom gesture dataset creation and training

## Technical Architecture

### Core Components
- **CNN Model**: Custom architecture for gesture recognition
- **OpenCV Pipeline**: Real-time hand detection and segmentation
- **PyQt6 Interface**: User-friendly GUI for interaction
- **SQLite Database**: Efficient gesture data management

### Model Architecture
- Convolutional Neural Network with multiple conv layers
- MaxPooling for feature extraction
- Dense layers with dropout for robust classification
- Input shape: 64x64 grayscale images
- Real-time inference capabilities

## Quick Start

The application is designed to be simple to use. Just run:
```bash
python interface.py
```
This launches the main interface where you can access all functionality.

### Complete Setup

1. Clone the repository:
```bash
git clone https://github.com/Mehdi-Soufiane/Listen_To_Me
cd Listen_To_Me
```

2. Install dependencies:
```bash
python -m pip install -r Install_Packages.txt
# For GPU acceleration:
python -m pip install -r Install_Packages_gpu.txt
```

3. Create your dataset:
- Run `python set_hand_histogram.py` for hand calibration
- Use `python create_gestures.py` to record your gestures
- Train the model with `python cnn_model_train.py`

## Dataset Creation Guide

1. **Hand Calibration**:
   - Run `set_hand_histogram.py`
   - Place your hand in the green squares
   - Press 'C' to capture
   - Press 'S' to save calibration

2. **Recording Gestures**:
   - Run `create_gestures.py`
   - Enter gesture name when prompted
   - Record variations of the gesture
   - Repeat for all desired gestures

3. **Model Training**:
   - After collecting all gestures
   - System will offer to train the model
   - Choose 'Yes' to start training

## Features

### Core Functionality
- Real-time sign language recognition
- Voice-to-text conversion
- Text-to-speech feedback
- Calculator mode using gestures

### Technical Features
- Custom dataset creation
- Real-time hand segmentation
- Model retraining capability
- Multi-modal interaction

## Development Stack

- Python 3.8+
- TensorFlow/Keras
- OpenCV
- PyQt6
- SQLite

## Future Enhancements

- Cloud model deployment
- Extended gesture vocabulary
- Enhanced model architecture
- Real-time feedback mechanism
- Support for additional sign languages

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgments

- MoroccoAI for organizing the InnovAI Hackathon
- Our mentors and advisors
- The deaf community for their invaluable feedback
- Open source communities for various tools and libraries used

## Contact

Project Link: https://github.com/Mehdi-Soufiane/Listen_To_Me

---

<p align="center">Made with ❤️ for accessibility and inclusion</p>
