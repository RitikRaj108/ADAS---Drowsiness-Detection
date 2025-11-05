import cv2
import mediapipe as mp
import numpy as np
from pygame import mixer
import time

# ===================== SETUP =====================
# Initialize pygame mixer for alarm
mixer.init()
mixer.music.load("alarm.wav")  # Make sure alarm.wav is in the same folder

# EAR threshold and frame count to trigger drowsiness
EAR_THRESHOLD = 0.25
CLOSED_EYES_FRAME_LIMIT = 30  # Number of consecutive frames

# MediaPipe eye landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# ===================== EAR CALCULATION =====================
def calculate_ear(eye_landmarks, frame_width, frame_height):
    points = [(int(l.x * frame_width), int(l.y * frame_height)) for l in eye_landmarks]

    # Use Euclidean distance
    A = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    B = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    C = np.linalg.norm(np.array(points[0]) - np.array(points[3]))

    ear = (A + B) / (2.0 * C)
    return ear

# ===================== FACE MESH INIT =====================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Open camera
cap = cv2.VideoCapture(0)

# Track state
closed_frames = 0
alarm_on = False

# ===================== MAIN LOOP =====================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get landmarks for both eyes
            left_eye = [face_landmarks.landmark[i] for i in LEFT_EYE]
            right_eye = [face_landmarks.landmark[i] for i in RIGHT_EYE]

            # Compute EAR for both eyes
            left_ear = calculate_ear(left_eye, w, h)
            right_ear = calculate_ear(right_eye, w, h)
            avg_ear = (left_ear + right_ear) / 2.0

            # Display EAR
            cv2.putText(frame, f'EAR: {avg_ear:.2f}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 255), 2)

            # Drowsiness detection logic
            if avg_ear < EAR_THRESHOLD:
                closed_frames += 1
                if closed_frames >= CLOSED_EYES_FRAME_LIMIT:
                    if not alarm_on:
                        mixer.music.play(-1)  # Loop the alarm
                        alarm_on = True
                    cv2.putText(frame, "DROWSINESS ALERT!", (100, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            else:
                closed_frames = 0
                if alarm_on:
                    mixer.music.stop()
                    alarm_on = False

    # Display video
    cv2.imshow("Drowsiness Detector", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ===================== CLEANUP =====================
cap.release()
cv2.destroyAllWindows()
