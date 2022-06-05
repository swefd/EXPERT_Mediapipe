import cv2
import mediapipe as mp

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


cap = cv2.VideoCapture(0)

mpDraw = mp.solutions.drawing_utils
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

            diff = dots[14][1] - dots[13][1]

            cv2.circle(frame, (dots[13][0], dots[13][1]), 3, (0, 255, 0), 2)
            cv2.circle(frame, (dots[14][0], dots[14][1]), 3, (0, 255, 0), 2)
            cv2.line(frame, (dots[13][0], dots[13][1]), (dots[14][0], dots[14][1]), (0, 255, 0), 2)

            cv2.putText(frame, str(13), (dots[13][0], dots[13][1]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
            cv2.putText(frame, str(14), (dots[14][0], dots[14][1]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
            cv2.putText(frame, str(diff), (dots[13][0] - 100, dots[13][1]), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 1)



    cv2.imshow("RES", frame)
    cv2.waitKey(1)