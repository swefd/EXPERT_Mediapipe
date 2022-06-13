import cv2
import mediapipe as mp
import math
import pyautogui

cap = cv2.VideoCapture(0)
cap.set(3, 1080)
cap.set(4, 720)

cv2.namedWindow("T-Rex Controller", cv2.WINDOW_NORMAL)

mpDraw = mp.solutions.drawing_utils
drawSpec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=2, color=(161, 159, 154))

mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh()

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX

mask = False
ruler = False
controller = False


def diff_points_counter(img, pt, id1=0, id2=0, point=(0, 0)):
    dots = {}
    for idx, lm in enumerate(pt.landmark):
        x, y = (int(lm.x * width), int(lm.y * height))
        dots[idx] = [x, y, idx]
    if point[0] or point[1]:
        xb = dots[id2][0]
        yb = point[1]
    else:
        xb = dots[id2][0]
        yb = dots[id2][1]
    xa = dots[id1][0]
    ya = dots[id1][1]
    line_center = (int((xb + xa) / 2), int((yb + ya) / 2))
    difference = int(math.sqrt((xb - xa) ** 2 + (yb - ya) ** 2))
    print(difference)
    cv2.line(img, (xb, yb), (xa, ya), GREEN, 2)
    cv2.putText(img, str(difference), line_center, FONT, 0.5, WHITE, 1)
    cv2.putText(img, str(id1), (xa, ya), FONT, 0.5, WHITE, 1)
    cv2.putText(img, str(id2), (xb, yb), FONT, 0.5, WHITE, 1)
    cv2.circle(img, (xa, ya), 2, RED, 2)
    cv2.circle(img, (xb, yb), 2, RED, 2)
    return difference


def draw_face_mask(img, pt):
    mpDraw.draw_landmarks(img, pt, mpFaceMesh.FACEMESH_TESSELATION, drawSpec, drawSpec)
    for idx, lm in enumerate(pt.landmark):
        x, y = (int(lm.x * width), int(lm.y * height))
        cv2.putText(img, str(idx), (x, y), FONT, 0.4, GREEN)


def game_controller(command):
    if command == "Jump":
        pyautogui.keyDown('space')
        print("Jump")
    elif command == "Fall":
        pyautogui.keyDown('down')
        print("Fall")
    else:
        pyautogui.keyUp('down')


while True:
    frame = cap.read()[1]
    height, width, deep = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(rgb)
    if results.multi_face_landmarks:
        for pts in results.multi_face_landmarks:
            key = cv2.waitKey(1)
            if key == ord("m"):
                mask = not mask
            if mask:
                draw_face_mask(frame, pts)
            if key == ord("d"):
                ruler = not ruler
            if ruler:
                diff_points_counter(frame, pts, 10, point=(540, 720))
                # diff_points_counter(frame, pts, 13, 14)
            if key == ord("p"):
                controller = not controller
            if controller:
                distance = diff_points_counter(frame, pts, 13, 14)
                if distance > 35:
                    game_controller("Jump")
                distance = diff_points_counter(frame, pts, 10, point=(540, 720))
                if distance < 490:
                    game_controller("Fall")
                else:
                    game_controller("Idle")
    cv2.putText(frame, "press m to open a mask or press d to open distance a ruler", (10, 30), FONT, 0.6, WHITE, 2)
    cv2.imshow("T-Rex Controller", frame)
    cv2.waitKey(1)
