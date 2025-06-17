import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Constants
CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
IMG_SIZE = (64, 64)
MODEL_PATH = 'asl_model.h5'
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence to consider a detection valid

def preprocess_image(img, target_size=IMG_SIZE):
    """Preprocess image for model input"""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the largest contour (hand)
        hand_contour = max(contours, key=cv2.contourArea)
        
        # Check if the contour is large enough to be a hand
        if cv2.contourArea(hand_contour) < 500:  # Lowered minimum area threshold
            return None
        
        # Create mask
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [hand_contour], -1, 255, -1)
        
        # Bitwise AND
        segmented = cv2.bitwise_and(gray, gray, mask=mask)
        
        # Resize and normalize
        resized = cv2.resize(segmented, target_size)
        normalized = resized / 255.0
        normalized = np.expand_dims(normalized, axis=-1)  # Add channel dimension
        
        # Show the thresholded/segmented image for debugging
        debug_img = cv2.resize(segmented, (200, 200))
        cv2.imshow('Preprocessed Hand (Debug)', debug_img)
        return normalized
    return None

def load_dataset(data_path, classes=CLASSES, test_size=0.2):
    """Load and preprocess dataset"""
    X = []
    y = []
    
    print("Starting to load dataset...")
    total_classes = len(classes)
    
    for label, class_name in enumerate(classes):
        class_dir = os.path.join(data_path, class_name)
        print(f"Processing class {class_name} ({label + 1}/{total_classes})...")
        
        # Skip if directory doesn't exist (for detection-only mode)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} not found, skipping...")
            continue
            
        image_count = 0
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            
            if img is not None:
                processed_img = preprocess_image(img)
                X.append(processed_img)
                y.append(label)
                image_count += 1
                
                if image_count % 100 == 0:
                    print(f"Processed {image_count} images for class {class_name}")
        
        print(f"Completed class {class_name}: {image_count} images processed")
    
    if not X:  # No data loaded
        print("Error: No images were loaded from the dataset!")
        return None, None, None, None
    
    print(f"\nTotal images loaded: {len(X)}")
    X = np.array(X)
    y = np.array(y)
    
    # One-hot encode labels
    y = to_categorical(y, num_classes=len(classes))
    
    print("Splitting data into training and testing sets...")
    return train_test_split(X, y, test_size=test_size, random_state=42)

def create_model(input_shape, num_classes=len(CLASSES)):
    """Create CNN model"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def train_model(data_path='asl_alphabet_train'):
    """Train the ASL recognition model"""
    print("\n=== Starting ASL Model Training ===")
    
    # Load dataset
    print("\nStep 1: Loading and preprocessing dataset...")
    X_train, X_test, y_train, y_test = load_dataset(data_path)
    
    if X_train is None:
        print(f"Error: No training data found in {data_path}")
        print("Please download the dataset from Kaggle: https://www.kaggle.com/datasets/grassknoted/asl-alphabet")
        print("And place the 'asl_alphabet_train' folder in the same directory as this script.")
        return None
    
    print(f"\nDataset loaded successfully!")
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    # Create model
    print("\nStep 2: Creating CNN model...")
    model = create_model(X_train.shape[1:])
    model.summary()
    
    # Train model
    print("\nStep 3: Training model...")
    print("This will take some time. Please wait...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=10,
        batch_size=32,
        verbose=1
    )
    
    # Save model
    print("\nStep 4: Saving model...")
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    return model

def detect_sign_language(model_path=MODEL_PATH):
    """Real-time ASL alphabet detection from webcam"""
    # Load model
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train the model first.")
        return
    
    model = load_model(model_path)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # ROI parameters
    roi_x, roi_y = 100, 100
    roi_size = 300
    
    print("\n=== ASL Alphabet Detection Started ===")
    print("Place your hand in the green box to detect ASL alphabets")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame horizontally
        frame = cv2.flip(frame, 1)
        
        # Draw ROI rectangle
        cv2.rectangle(frame, (roi_x, roi_y), 
                     (roi_x + roi_size, roi_y + roi_size), 
                     (0, 255, 0), 2)
        
        # Extract ROI
        roi = frame[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
        
        # Preprocess ROI
        processed_roi = preprocess_image(roi)
        
        if processed_roi is not None:
            # Predict
            prediction = model.predict(np.array([processed_roi]), verbose=0)
            confidence = np.max(prediction)
            predicted_class = CLASSES[np.argmax(prediction)]
            
            # Only show detection if confidence is high enough
            if confidence > CONFIDENCE_THRESHOLD:
                # Display current detection
                cv2.putText(frame, f"Detected: {predicted_class} ({confidence:.2f})", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No clear sign detected", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "No hand detected", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show frame
        cv2.imshow("ASL Alphabet Detection", frame)
        
        # Handle key presses
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyWindow('Preprocessed Hand (Debug)')
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ASL Alphabet Recognition System")
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--detect', action='store_true', help='Run real-time detection')
    args = parser.parse_args()
    
    if args.train:
        train_model()
    elif args.detect:
        detect_sign_language()
    else:
        print("Please specify either --train or --detect flag")
        print("Example:")
        print("  python asl_detector.py --train  # To train the model")
        print("  python asl_detector.py --detect # To run real-time detection")