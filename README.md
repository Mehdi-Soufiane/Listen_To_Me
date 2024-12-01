# Listen To Me - Sign Language Recognition System

> Fast communication, Easy life - Breaking barriers in deaf communication through AI

[![Flutter Version](https://img.shields.io/badge/Flutter-3.0-blue.svg)](https://flutter.dev/)
[![Python Version](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## About The Project

Listen To Me is an innovative sign language recognition system developed during the MoroccoAI InnovAI Hackathon 2024. The project aims to bridge the communication gap for over 466 million people worldwide with disabling hearing loss, including 300,000 individuals in Morocco.

### The Problem
- Communication barrier for deaf individuals worldwide
- Limited access to real-time translation between sign language and spoken language
- Lack of seamless integration between deaf and hearing communities

### Our Solution
Our application provides:
- Real-time sign language recognition using computer vision
- Voice-to-text conversion for accessibility
- Text-to-speech functionality
- Intuitive mobile interface
- High accuracy gesture recognition (>95% accuracy)

## Technical Architecture

### Frontend
- Flutter for cross-platform mobile development
- Material Design UI components
- Real-time camera integration
- Speech-to-text and text-to-speech modules

### Backend
- Python-based REST API using Flask
- TensorFlow/Keras for deep learning models
- OpenCV for image processing
- SQLite database for gesture storage
- PyQt6 for desktop interface

### Machine Learning Pipeline
- CNN architecture for gesture recognition
- Real-time hand detection and segmentation
- Data augmentation for improved model robustness
- Transfer learning for optimized performance

## Getting Started

### Prerequisites
```bash
# Install Python packages
python -m pip install -r Install_Packages.txt

# For GPU support
python -m pip install -r Install_Packages_gpu.txt
```

### Installation & Setup

1. Clone the repository:
```bash
git clone [repository-url]
cd listen-to-me
```

2. Set up the hand histogram for gesture detection:
```bash
python set_hand_histogram.py
```

3. Create your gesture dataset:
```bash
python create_gestures.py
```

4. Train the model:
```bash
python cnn_model_train.py
```

5. Launch the application:
```bash
python interface.py
```

## Usage Guide

### 1. Calibration
- Run hand calibration first
- Follow on-screen instructions
- Place hand in green squares
- Press 'C' to capture
- Press 'S' to save

### 2. Creating Gestures
- Launch gesture creation tool
- Enter gesture name
- Record multiple variations
- Save and train model

### 3. Recognition
- Start recognition mode
- Choose between text or calculator mode
- Use voice commands if needed
- View real-time translations

## Features

- Real-time sign language recognition
- Voice-to-text conversion
- Text-to-speech feedback
- Calculator mode using gestures
- Gesture visualization tools
- Custom gesture training
- Multi-language support
- Intuitive GUI interface

## Project Structure
```
.
├── dataset/                # Gesture image datasets
├── cnn_model_train.py     # CNN model training
├── create_gestures.py     # Dataset creation tool
├── display_gestures.py    # Visualization utility
├── final.py              # Main recognition script
├── interface.py          # GUI application
├── load_images.py        # Image processing
├── Rotate_images.py      # Data augmentation
└── set_hand_histogram.py # Calibration tool
```

## Development Stack

- Python 3.8+
- TensorFlow/Keras
- OpenCV
- PyQt6
- Flutter/Dart
- SQLite
- Flask RESTful API

## Future Enhancements

- Cloud deployment
- API development for third-party integration
- Extended gesture vocabulary
- Improved model accuracy
- Mobile app optimization
- Real-time feedback mechanism
- Support for additional sign languages

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Team

Project developed during MoroccoAI InnovAI Hackathon 2024 by:
- [Team Member 1](https://github.com/member1)
- [Team Member 2](https://github.com/member2)
- [Team Member 3](https://github.com/member3)

## Acknowledgments

- MoroccoAI for organizing the InnovAI Hackathon
- Our mentors and advisors
- The deaf community for their invaluable feedback
- Open source communities for various tools and libraries used

## Contact

Project Link: [https://github.com/yourusername/listen-to-me](https://github.com/yourusername/listen-to-me)

---

<p align="center">Made with ❤️ for accessibility and inclusion</p>
