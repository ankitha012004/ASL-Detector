# ASL Alphabet Recognition System

This project is a real-time American Sign Language (ASL) alphabet recognition system using a webcam and a deep learning model.

## Features
- Detects ASL alphabets (A-Y, except J and Z) in real-time from webcam video
- Uses a trained Convolutional Neural Network (CNN)
- Simple and easy-to-use interface

## Setup Instructions

1. **Clone the repository**
   ```sh
   git clone <your-repo-url>
   cd SIGN LANGUAGE CONVERTER
   ```

2. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```

3. **Download the Model and Dataset**
   - Download the trained model file [`asl_model.h5`] from your cloud storage (Google Drive, Dropbox, etc.) and place it in the project root.
   - Download the ASL Alphabet dataset from [Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) if you want to retrain the model. Place the `asl_alphabet_train` folder in the project root.

4. **Run the Detector**
   ```sh
   python asl_detector.py --detect
   ```

5. **(Optional) Train the Model**
   ```sh
   python asl_detector.py --train
   ```

## Notes
- The model and dataset are **not included** in this repository due to their large size. Please download them separately as described above.
- Make sure your webcam is connected and working.
- For best results, use a plain background and good lighting.

## License
This project is for educational purposes. 
