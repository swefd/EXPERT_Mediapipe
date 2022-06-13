import cv2
import mediapipe as mp
import pyautogui

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
    cv2.putText(frame, str(diff), (dots[13][0] + 150, int((dots[13][1] + dots[14][1]) / 2)),
                cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

    faceDiff = diff_points_counter(dots[152][1], dots[10][1])
    cv2.putText(frame, str(faceDiff), (dots[13][0] - 200, int((dots[1][1] + dots[152][1]) / 2)),
                cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

    # print("Lips diff: {0}".format(diff))
    # print("Face diff: {0}".format(faceDiff - diff))


def check_action():
    mDif = diff_points_counter(dots[13][1], dots[14][1])
    mTriggerDif = diff_points_counter(dots[10][1], dots[152][1]) / 10

    fDif_W = diff_points_counter(dots[234][0], dots[454][0])
    fDif_W += fDif_W / 8
    fDif_H = diff_points_counter(dots[10][1], dots[152][1])

    if mDif > mTriggerDif:
        print("Jump")
        pyautogui.keyDown('space')

    if fDif_H < fDif_W:
        pyautogui.keyDown('down')
    else:
        pyautogui.keyUp('down')


    # if fDif > fTriggerDif:
    #     pyautogui.keyDown('down')
    # else:
    #     pyautogui.keyUp('down')

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
            check_action()

    cv2.imshow("RES", frame)
    cv2.waitKey(1)
