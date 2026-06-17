import cv2
import numpy as np
import time
from pyzbar.pyzbar import decode
from ultralytics import YOLO
from deepface import DeepFace

# --- FEATURE : Liveness Detection  ---
def is_live_face(face_roi):
    """
    Checks texture variance using Laplacian. 
    Printed photos/screens usually have lower high-frequency details (blurrier).
    """
    laplacian_var = cv2.Laplacian(face_roi, cv2.CV_64F).var()
    if laplacian_var < 50 or laplacian_var > 520:
        return False # Fake
    return True # Real 3D Face

def main():
    print("=== INITIALIZING AI SECURITY SYSTEM ===")
    
    # Load LBPH Face Recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer.yml')
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # --- FEATURE : Load YOLOv8 for Threat Detection ---
    print("[INFO] Loading YOLOv8 Model...")
    yolo_model = YOLO('yolov8n.pt') 

    camera = cv2.VideoCapture(0)
    frame_counter = 0
    current_emotion = "Neutral"

    while True:
        ret, frame = camera.read()
        if not ret:
            break
        
        frame_counter += 1
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        enhanced_gray = cv2.equalizeHist(gray_frame) # Syllabus: Filtering

        # Variables for authentication
        qr_detected = False
        expected_id = -1
        expected_name = "Unknown"

        # --- 1. QR CODE DETECTION & HOMOGRAPHY ---
        qr_codes = decode(enhanced_gray)
        for qr in qr_codes:
            qr_detected = True
            qr_data = qr.data.decode('utf-8')
            
            try:
                # Format: ID:1-John-168538392
                expected_id = int(qr_data.split('-')[0].split(':')[1])
                expected_name = qr_data.split('-')[1]
            except:
                expected_id = -1

            pts = np.array([qr.polygon], np.int32)
            cv2.polylines(frame, [pts], True, (255, 255, 0), 3)

        # --- 2. THREAT DETECTION (YOLOv8) ---
        weapon_detected = False
        if frame_counter % 3 == 0:
            yolo_results = yolo_model(frame, stream=True, verbose=False)
            for r in yolo_results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id == 76: 
                        weapon_detected = True
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(frame, "WEAPON DETECTED!", (x1, y1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

        # --- 3. FACE RECOGNITION & LIVENESS & EMOTION ---
        faces = face_detector.detectMultiScale(enhanced_gray, 1.3, 5)

        if frame_counter % 2 == 0:

            for (x, y, w, h) in faces:
                face_roi_gray = enhanced_gray[y:y+h, x:x+w]
                face_roi_color = frame[y:y+h, x:x+w]

                # A. Liveness Check 
                is_real = is_live_face(face_roi_gray)

                # B. Emotion Recognition 
                if frame_counter % 30 == 0 and is_real:
                    try:
                        # Convert to RGB for DeepFace
                        rgb_face = cv2.cvtColor(face_roi_color, cv2.COLOR_BGR2RGB)
                        analysis = DeepFace.analyze(rgb_face, actions=['emotion'], enforce_detection=False)
                        current_emotion = analysis[0]['dominant_emotion']
                    except:
                        pass

                # C. Identity Prediction
                predicted_id, confidence = recognizer.predict(face_roi_gray)

                if weapon_detected:
                    color = (0, 0, 255)
                    status_text = "SYSTEM LOCKED: THREAT"
                elif not is_real:
                    color = (0, 0, 255)
                    status_text = "DENIED: FAKE FACE"
                elif qr_detected:
                    if predicted_id == expected_id and confidence < 75:
                        color = (0, 255, 0)
                        status_text = f"GRANTED: {expected_name}"
                    else:
                        color = (0, 0, 255)
                        status_text = "DENIED: MISMATCH"
                else:
                    color = (0, 165, 255)
                    status_text = "WAITING FOR ID CARD..."

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, status_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Display Emotion & Panic Alarm
                cv2.putText(frame, f"Emotion: {current_emotion.upper()}", (x, y + h + 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                if current_emotion in ['fear', 'sad', 'angry']:
                    cv2.putText(frame, "SILENT PANIC ALARM TRIGGERED", (20, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

            cv2.imshow('AI Security Feed', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()