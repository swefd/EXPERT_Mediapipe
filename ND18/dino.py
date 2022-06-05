import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh()


def diff_points_counter(point1, point2):
    return abs(point1 - point2)

def draw_face_mask(dots):
    cv2.circle(frame, (dots[13][0], dots[13][1]), 2, (0, 0, 255), 2)
    cv2.circle(frame, (dots[14][0], dots[14][1]), 2, (0, 0, 255), 2)

    cv2.line(frame, (dots[13][0], dots[13][1]), (dots[14][0], dots[14][1]), (0, 255, 0), 2)

    diff = diff_points_counter(dots[14][1], dots[13][1])
    cv2.putText(frame, str(diff), (dots[13][0], int((dots[13][1] + dots[14][1]) / 2)),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)

    faceDiff = diff_points_counter(dots[152][1], dots[10][1])
    cv2.putText(frame, str(faceDiff), (dots[13][0] - 100, int((dots[1][1] + dots[152][1]) / 2)),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)

    print("Lips diff: {0}".format(diff))
    print("Face diff: {0}".format(faceDiff - diff))


while True:
    frame = cap.read()[1]
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(frameRGB)

    if results.multi_face_landmarks:
        for faces in results.multi_face_landmarks:
            dots = {}
            for idx, lm in enumerate(faces.landmark):
                height, width, deep = frame.shape

                x = int(lm.x * width)
                y = int(lm.y * height)
                dots[idx] = [x, y, idx]

            draw_face_mask(dots)





    cv2.imshow("RES", frame)
    cv2.waitKey(1)
