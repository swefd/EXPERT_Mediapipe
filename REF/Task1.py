import cv2
import mediapipe as mp

# pip install 'protobuf~=3.19.0'

cap = cv2.VideoCapture(0)

mpDraw = mp.solutions.drawing_utils
drawStyles = mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
circleDrawingSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))
lineDrawingSpec = mpDraw.DrawingSpec(thickness=1, color=(255, 255, 0))

mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh()

while True:
   frame = cap.read()[1]
   rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
   results = faceMesh.process(rgb)

   if results.multi_face_landmarks:
       for faces in results.multi_face_landmarks:
           mpDraw.draw_landmarks(frame, faces, mpFaceMesh.FACEMESH_TESSELATION, circleDrawingSpec, lineDrawingSpec)

   cv2.imshow("frame", frame)
   cv2.waitKey(1)