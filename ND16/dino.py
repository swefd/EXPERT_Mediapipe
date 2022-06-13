import cv2
import mediapipe as mp
import pyautogui

mask = False
ruler = False
controller = False

cap = cv2.VideoCapture(0)

mpDraw = mp.solutions.drawing_utils
drawSpec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=2, color=(161, 159, 154))
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh()


def diff_points_counter(point1, point2):
    return abs(point1 - point2)


def draw_face_mask(dots):
    if mask == True:
        cv2.circle(frame, (dots[13][0], dots[13][1]), 2, (0, 0, 255), 2)
        cv2.circle(frame, (dots[14][0], dots[14][1]), 2, (0, 0, 255), 2)
        cv2.line(frame, (dots[13][0], dots[13][1]), (dots[14][0], dots[14][1]), (0, 255, 0), 2)

    if ruler:
        diff = diff_points_counter(dots[14][1], dots[13][1])
        cv2.putText(frame, str(diff), (dots[13][0], int((dots[13][1] + dots[14][1]) / 2)),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
        faceDiff = diff_points_counter(dots[152][1], dots[10][1])
        cv2.putText(frame, str(faceDiff), (dots[13][0] - 100, int((dots[1][1] + dots[152][1]) / 2)),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)

    # print("Lips diff: {0}".format(diff))
    # print("Face diff: {0}".format(faceDiff - diff))


def check_actions():
    pointUpL = dots[13][1]
    pointDownL = dots[14][1]

    pointUpF = dots[10][1]
    pointDownF = dots[164][1]

    mDif = diff_points_counter(dots[10][1], dots[152][1]) / 10

    if diff_points_counter(pointUpL, pointDownL) > mDif:
        pyautogui.keyDown('space')
    if diff_points_counter(pointUpF, pointDownF) > 300:
        pyautogui.keyDown('down')
    else:
        pyautogui.keyUp('down')


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

        key = cv2.waitKey(1)
        if key == ord("m"):
            mask = not mask
            print("MASK IS: " + str(mask))
        if key == ord("d"):
            ruler = not ruler
            print("RULLER IS: " + str(ruler))
        if key == ord("p"):
            controller = not controller
            print("GAME MODE IS: " + str(controller))

        draw_face_mask(dots)

        if controller:
            check_actions()

    cv2.imshow("Res", frame)

