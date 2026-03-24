import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

if face_cascade.empty():
    print("❌ Model not loaded")
else:
    print("✅ Model loaded")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]

        # Feature 1: Brightness
        # Feature 1: Brightness
        brightness = np.mean(face)

        # Feature 2: Face size (movement proxy)
        face_area = w * h

        # Improved logic
        if brightness > 160:
            emotion = "Happy 😊"
        elif brightness < 70:
            emotion = "Sad 😔"
        elif face_area > 80000:
            emotion = "Confident 😎"
        else:
            emotion = "Neutral 😐"

        # Draw box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.putText(frame, f"Emotion: {emotion}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2)

    cv2.imshow("Emotion Detection (AI)", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()