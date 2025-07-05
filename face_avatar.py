import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Basic function to estimate emotion using lip landmarks
def estimate_emotion(landmarks):
    if landmarks:
        mouth_open = landmarks[14][1] - landmarks[13][1]
        if mouth_open > 20:
            return "Surprised", (0, 255, 255)
        elif landmarks[159][1] - landmarks[145][1] > 5:
            return "Happy", (0, 255, 0)
        else:
            return "Neutral", (255, 255, 255)
    return "Unknown", (128, 128, 128)

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        emotion_label = "None"
        emotion_color = (255, 255, 255)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

                h, w, _ = frame.shape
                landmarks = [(int(pt.x * w), int(pt.y * h)) for pt in face_landmarks.landmark]
                emotion_label, emotion_color = estimate_emotion(landmarks)

        # Draw Emotion Avatar
        cv2.putText(frame, f'Emotion: {emotion_label}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, emotion_color, 2)
        cv2.circle(frame, (550, 100), 40, emotion_color, -1)
        cv2.putText(frame, emotion_label, (500, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, emotion_color, 2)

        cv2.imshow('Real-Time Emotion Avatar', frame)

        if cv2.waitKey(5) & 0xFF == 27:  # ESC key
            break

cap.release()
cv2.destroyAllWindows()
