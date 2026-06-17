import cv2
import numpy as np
import os

def train_system():
    print("=== BIOMETRIC SYSTEM TRAINING ===")
    
    # Path for face image database
    dataset_path = 'dataset'
    
    # Initialize LBPH Face Recognizer (Local Binary Patterns Histograms)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    # Initialize Haar Cascade to verify faces before training
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def get_images_and_labels(path):
        # Get all file paths in the dataset folder
        image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
        
        face_samples = []
        user_ids = []

        for image_path in image_paths:
            # Read the image in grayscale mode
            gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            # Extract the user ID from the file name (e.g., "User.1.25.jpg" -> ID is 1)
            file_name = os.path.split(image_path)[-1]
            user_id = int(file_name.split(".")[1])
            
            # Detect faces in the current image (to ensure we only train on the face)
            faces = detector.detectMultiScale(gray_img)
            
            for (x, y, w, h) in faces:
                # Append the cropped face and its corresponding ID to the lists
                face_samples.append(gray_img[y:y+h, x:x+w])
                user_ids.append(user_id)
                
        return face_samples, user_ids

    print("[INFO] Training faces. This might take a few seconds...")
    faces, ids = get_images_and_labels(dataset_path)
    
    # Train the model
    recognizer.train(faces, np.array(ids))

    # Save the trained model into a file named 'trainer.yml'
    recognizer.write('trainer.yml') 
    
    # Print the number of unique faces trained
    unique_faces = len(np.unique(ids))
    print(f"[SUCCESS] {unique_faces} unique face(s) trained and saved to 'trainer.yml'.")

if __name__ == "__main__":
    train_system()