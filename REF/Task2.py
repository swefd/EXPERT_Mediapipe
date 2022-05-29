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
           mpDraw.draw_landmarks(frame, faces, mpFaceMesh.FACEMESH_TESSELATION, drawSpec, drawSpec)
           for id, lm in enumerate(faces.landmark):
               print(f"x:{lm.x},y:{lm.y},z:{lm.z} index: {id}")
               height, width, deep = frame.shape
               x, y = (int(lm.x * width), int(lm.y * height))
               cv2.putText(frame, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)

   cv2.imshow("frame", frame)
   cv2.waitKey(1)