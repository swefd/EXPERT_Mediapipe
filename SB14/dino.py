import cv2
import mediapipe as mp
import pyautogui

playGame = False
drawFaceMask = False

cap = cv2.VideoCapture(0)

mpDraw = mp.solutions.drawing_utils
drawSpec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=2, color=(161, 159, 154))
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh()


def diff_points_counter(a, b):
    return abs(a - b)


def draw_face_mask(img):
    cv2.circle(img, (dots[13][0], dots[13][1]), 2, (255, 0, 0), 5)
    cv2.circle(img, (dots[14][0], dots[14][1]), 2, (255, 0, 0), 5)
    cv2.putText(img, str(diff_points_counter(dots[14][1], dots[13][1])),
                (200, dots[14][1]), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    cv2.line(img, (dots[13][0], dots[13][1]), (dots[14][0], dots[14][1]), (0, 0, 255), 2)


def check_action():
    L_upperPoint = dots[13][1]
    L_lowerPoint = dots[14][1]

    F_upperPoint = dots[10][1]
    F_lowerPoint = dots[164][1]

    print(diff_points_counter(F_upperPoint, F_lowerPoint))

    if diff_points_counter(L_upperPoint, L_lowerPoint) > 50:
        print("Jump")
        pyautogui.keyDown("space")

    if diff_points_counter(F_upperPoint, F_lowerPoint) > 300:
        print("DOWN")
        pyautogui.keyDown("down")
    else:
        pyautogui.keyUp("down")


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

            if key == ord('p'):
                playGame = not playGame
                print("GAME MODE IS: " + str(playGame))
            elif key == ord('m'):
                drawFaceMask = not drawFaceMask
                print("FaceMask IS: " + str(drawFaceMask))

            if playGame:
                check_action()
            if drawFaceMask:
                draw_face_mask(frame)

    cv2.imshow("Res", frame)
    cv2.waitKey(1)
