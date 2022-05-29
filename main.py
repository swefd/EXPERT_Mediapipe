import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mpDraw = mp.solutions.drawing_utils
drawStyles = mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
testStyle = mpDraw.DrawingSpec(thickness=1, color=(0, 255, 0))
testStyle2 = mpDraw.DrawingSpec(thickness=1, color=(0, 255, 255))

mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh()


while True:
    frame = cap.read()[1]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(rgb)

    if results.multi_face_landmarks:
        for faces in results.multi_face_landmarks:
            mpDraw.draw_landmarks(frame, faces, mpFaceMesh.FACEMESH_TESSELATION, testStyle, testStyle2)

    cv2.imshow("Frame", frame)
    cv2.waitKey(1)