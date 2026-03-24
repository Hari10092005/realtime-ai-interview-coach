import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd

# -----------------------
# INIT
# -----------------------
mp_face_mesh = mp.solutions.face_mesh
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

blink_count = 0
blink_flag = False
drowsy_frames = 0
voice_result = "Not tested"
emotion_history = []

# -----------------------
# VOICE FUNCTION
# -----------------------
def analyze_voice():
    global voice_result

    fs = 44100
    duration = 3

    print("🎤 Speak now...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()

    audio = audio.flatten()

    volume = np.mean(np.abs(audio))
    variation = np.std(audio)

    if volume > 0.08 and variation > 0.04:
        voice_result = "Stressed 😰"
    elif volume > 0.03:
        voice_result = "Confident 😎"
    else:
        voice_result = "Nervous 😬"

# -----------------------
# CAMERA
# -----------------------
cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(refine_landmarks=True) as face_mesh:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        results = face_mesh.process(rgb)

        # -----------------------
        # FACE + EMOTION
        # -----------------------
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        emotion = "Neutral 😐"

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            brightness = np.mean(face)

            if brightness > 160:
                emotion = "Happy 😊"
            elif brightness < 70:
                emotion = "Sad 😔"
            else:
                emotion = "Neutral 😐"

            # Smooth emotion
            emotion_history.append(emotion)
            if len(emotion_history) > 10:
                emotion_history.pop(0)

            emotion = max(set(emotion_history), key=emotion_history.count)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # -----------------------
        # EYE TRACKING
        # -----------------------
        status = "Focused 👀"

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark

                def eye_ratio(eye):
                    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye]
                    return (abs(p2.y - p6.y) + abs(p3.y - p5.y)) / (2 * abs(p1.x - p4.x))

                LEFT = [33,160,158,133,153,144]
                RIGHT = [362,385,387,263,373,380]

                ear = (eye_ratio(LEFT) + eye_ratio(RIGHT)) / 2

                if ear < 0.23:
                    if not blink_flag:
                        blink_count += 1
                        blink_flag = True
                    drowsy_frames += 1
                else:
                    blink_flag = False
                    drowsy_frames = 0

                if drowsy_frames > 15:
                    status = "Drowsy 😴"
                else:
                    status = "Focused 👀"

        # -----------------------
        # UI PANEL
        # -----------------------
        # -----------------------
        # MODERN UI PANEL
        # -----------------------

        overlay = frame.copy()

        # Transparent panel (glass effect)
        cv2.rectangle(overlay, (0, 0), (400, 200), (30, 30, 30), -1)
        alpha = 0.6
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Title
        cv2.putText(frame, "AI INTERVIEW ANALYZER", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Divider line
        cv2.line(frame, (10, 35), (380, 35), (100, 100, 100), 1)

        # Status indicators
        cv2.putText(frame, f"Blinks   : {blink_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(frame, f"Status   : {status}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.putText(frame, f"Emotion  : {emotion}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        cv2.putText(frame, f"Voice    : {voice_result}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 255), 2)

        # Confidence bar
        confidence = min(100, int((blink_count + len(emotion_history)) * 2))

        # Bar background
        cv2.rectangle(frame, (10, 170), (210, 185), (50, 50, 50), -1)

        # Filled bar
        cv2.rectangle(frame, (10, 170), (10 + 2 * confidence, 185), (0, 255, 100), -1)

        cv2.putText(frame, f"{confidence}%", (220, 183),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Bottom instruction
        cv2.putText(frame, "Press V: Voice | ESC: Exit",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("AI Interview Analyzer - Live", frame)

        key = cv2.waitKey(1)

        if key == ord('v'):
            analyze_voice()

        if key == 27:
            break

cap.release()
cv2.destroyAllWindows()