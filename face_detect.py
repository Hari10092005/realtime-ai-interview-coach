import cv2
import mediapipe as mp

print("Starting Face Detection...")

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not working")
    exit()

print("✅ Camera started")

with mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
) as face_detection:

    while True:
        ret, frame = cap.read()

        if not ret:
            print("❌ Frame error")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(frame, detection)

                confidence = detection.score[0]
                cv2.putText(
                    frame,
                    f"Confidence: {int(confidence*100)}%",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(1) == 27:  # ESC
            print("Exiting...")
            break

cap.release()
cv2.destroyAllWindows()