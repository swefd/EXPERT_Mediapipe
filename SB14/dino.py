import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mpDraw = mp.solutions.drawing_utils
drawSpec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=2, color=(161, 159, 154))
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh()

while True:
    frame = cap.read()[1]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(rgb)
    if results.multi_face_landmarks:
        for faces in results.multi_face_landmarks:
            dots = {}
            for idx, lm in enumerate(faces.landmark):
                height, width, deep = frame.shape
                x = int(lm.x * width)
                y = int(lm.y * height)
                dots[idx] = [x, y, idx]

            cv2.circle(frame, (dots[13][0], dots[13][1]), 2, (0, 255, 0), 2)
            cv2.circle(frame, (dots[14][0], dots[14][1]), 2, (0, 255, 0), 2)
            cv2.putText(frame, str(dots[14][1] - dots[13][1]), (200, dots[14][1]), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            cv2.line(frame, (dots[13][0], dots[13][1]), (dots[14][0], dots[14][1]), (0, 255, 0), 3)


    cv2.imshow("Res", frame)
    cv2.waitKey(1)