import cv2
import os
import qrcode
import time
import sys

def main():
    print("=== BIOMETRIC SYSTEM REGISTRATION ===")
    
    # Bilgileri terminalden değil, Tkinter arayüzünden (sys.argv) alıyoruz
    if len(sys.argv) >= 3:
        user_id = sys.argv[1]
        user_name = sys.argv[2]
    else:
        print("[ERROR] Please run this via the main dashboard!")
        return

    folder = 'dataset'
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"[INFO] Created a new folder named '{folder}'.")

    # FEATURE 1: Prevent Duplicate Registration
    existing_files = os.listdir(folder)
    for file in existing_files:
        if file.startswith(f"User.{user_id}."):
            print(f"[ERROR] User ID {user_id} already exists! Please use a different ID.")
            return

    # FEATURE 6: Unique Dynamic QR Code (Added Timestamp)
    unique_timestamp = int(time.time())
    qr_data = f"ID:{user_id}-{user_name}-{unique_timestamp}"
    img_qr = qrcode.make(qr_data)
    qr_filename = f"{user_name}_QR_Card.png"
    img_qr.save(qr_filename)
    print(f"[INFO] Unique personalized QR code saved as '{qr_filename}'.")

    camera = cv2.VideoCapture(0)
    time.sleep(2.0) # Warm-up
    
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    count = 0
    print("[INFO] Starting camera... Please look at the camera.")

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        # FEATURE 2: Warn if no face is detected
        if len(faces) == 0:
            cv2.putText(frame, "WARNING: NO FACE DETECTED!", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            count += 1
            
            file_path = f"{folder}/User.{user_id}.{count}.jpg"
            cv2.imwrite(file_path, gray_frame[y:y+h, x:x+w])

        cv2.imshow('User Registration - (Press Q to exit)', frame)

        if cv2.waitKey(100) & 0xFF == ord('q') or count >= 50:
            break

    print(f"[SUCCESS] {count} face images captured.")
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()