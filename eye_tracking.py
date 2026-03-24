import cv2
import mediapipe as mp

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh

cap = cv2.VideoCapture(0)

# Eye landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Blink variables
blink_count = 0
blink_flag = False
drowsy_frames = 0


def eye_aspect_ratio(landmarks, eye):
    p1 = landmarks[eye[0]]
    p2 = landmarks[eye[1]]
    p3 = landmarks[eye[2]]
    p4 = landmarks[eye[3]]
    p5 = landmarks[eye[4]]
    p6 = landmarks[eye[5]]

    vertical1 = abs(p2.y - p6.y)
    vertical2 = abs(p3.y - p5.y)
    horizontal = abs(p1.x - p4.x)

    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear


with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:

                landmarks = face_landmarks.landmark

                # EAR calculation
                left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
                right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
                avg_ear = (left_ear + right_ear) / 2

                # -----------------------
                # BLINK + DROWSINESS
                # -----------------------
                if avg_ear < 0.25:
                    if not blink_flag:
                        blink_count += 1
                        blink_flag = True
                    drowsy_frames += 1
                else:
                    blink_flag = False
                    drowsy_frames = 0

                if drowsy_frames > 15:
                    status = "Drowsy 😴"
                elif avg_ear < 0.25:
                    status = "Blinking 👁️"
                else:
                    status = "Focused 👀"

                # -----------------------
                # EYE DIRECTION
                # -----------------------
                left_eye_x = landmarks[33].x

                if left_eye_x < 0.4:
                    direction = "Looking Left ⬅️"
                elif left_eye_x > 0.6:
                    direction = "Looking Right ➡️"
                else:
                    direction = "Looking Center 🎯"

                # -----------------------
                # DRAW LANDMARKS
                # -----------------------
                for lm in landmarks:
                    x = int(lm.x * frame.shape[1])
                    y = int(lm.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

                # -----------------------
                # DISPLAY TEXT
                # -----------------------
                cv2.putText(frame, f"Blinks: {blink_count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.putText(frame, status, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                cv2.putText(frame, direction, (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Advanced Eye Tracking", frame)

        if cv2.waitKey(1) == 27:
            break

cap.release()
cv2.destroyAllWindows()