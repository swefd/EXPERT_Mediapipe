import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh()

defDrawStyle = mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()

circleStyle = mp.solutions.drawing_utils.DrawingSpec(thickness=1, color=(0, 0, 255), circle_radius=1)
meshStyle = mp.solutions.drawing_utils.DrawingSpec(thickness=1, color=(0, 255, 0))

while True:
    frame = cap.read()[1]
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(frameRGB)

    if results.multi_face_landmarks:
        for faces in results.multi_face_landmarks:
            mpDraw.draw_landmarks(frame, faces, mpFaceMesh.FACEMESH_TESSELATION, circleStyle, meshStyle)

    cv2.imshow("RES", frame)
    cv2.waitKey(1)

