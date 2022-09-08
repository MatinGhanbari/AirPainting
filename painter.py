import cv2
import numpy as np
import time
import os

import track_hands as TH

brush_thickness = 15
eraser_thickness = 100
image_canvas = np.zeros((720, 1280, 3), np.uint8)

currentT = 0
previousT = 0

header_img = "Images"
header_img_list = os.listdir(header_img)
overlay_image = []

for i in header_img_list:
    image = cv2.imread(f'{header_img}/{i}')
    overlay_image.append(image)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

cap.set(cv2.CAP_PROP_FPS, 60)

default_overlay = overlay_image[0]
draw_color = (255, 200, 100)

detector = TH.handDetector(min_detection_confidence=.85)

xp = 0
yp = 0

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = detector.findHands(frame, draw=False)
    landmark_list = detector.findPosition(frame, draw=False)

    frame[0:125, 0:1280] = default_overlay

    if len(landmark_list) != 0:
        x1, y1 = (landmark_list[8][1:])  # index
        x2, y2 = landmark_list[12][1:]  # middle
        cv2.circle(frame, (x1, y1), 10, color=draw_color, thickness=-1)

        my_fingers = detector.fingerStatus()
        if my_fingers[1] and my_fingers[2]:
            cv2.circle(frame, (x2, y2), 10, color=draw_color, thickness=-1)
            xp, yp = 0, 0
            if y1 < 125:
                if 380 < x1 < 480:
                    default_overlay = overlay_image[0]
                    draw_color = (255, 0, 0)
                elif 480 < x1 < 630:
                    default_overlay = overlay_image[1]
                    draw_color = (47, 225, 245)
                elif 640 < x1 < 770:
                    default_overlay = overlay_image[2]
                    draw_color = (197, 47, 245)
                elif 770 < x1 < 910:
                    default_overlay = overlay_image[3]
                    draw_color = (53, 245, 47)
                elif 1065 < x1 < 1250:
                    default_overlay = overlay_image[4]
                    draw_color = (0, 0, 0)

            cv2.putText(frame, 'Color Selector Mode', (900, 700), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        color=(255, 255, 255), thickness=5, fontScale=1)
            cv2.putText(frame, 'Color Selector Mode', (900, 700), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        color=(179, 54, 236), thickness=2, fontScale=1)
            cv2.line(frame, (x1, y1), (x2, y2), color=draw_color, thickness=3)

        if my_fingers[1] and not my_fingers[2]:
            cv2.putText(frame, 'Writing Mode', (1000, 700), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        color=(255, 255, 255), thickness=5, fontScale=1)
            cv2.putText(frame, 'Writing Mode', (1000, 700), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        color=(179, 54, 236), thickness=2, fontScale=1)

            cv2.circle(frame, (x1, y1), 15, draw_color, thickness=-1)

            if xp == 0 and yp == 0:
                xp = x1
                yp = y1

            if draw_color == (0, 0, 0):
                cv2.line(frame, (xp, yp), (x1, y1), color=draw_color, thickness=eraser_thickness)
                cv2.line(image_canvas, (xp, yp), (x1, y1), color=draw_color, thickness=eraser_thickness)

            else:
                cv2.line(frame, (xp, yp), (x1, y1), color=draw_color, thickness=brush_thickness)
                cv2.line(image_canvas, (xp, yp), (x1, y1), color=draw_color, thickness=brush_thickness)

            xp, yp = x1, y1

    img_gray = cv2.cvtColor(image_canvas, cv2.COLOR_BGR2GRAY)
    _, imginv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    imginv = cv2.cvtColor(imginv, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, imginv)
    frame = cv2.bitwise_or(frame, image_canvas)
    currentT = time.time()
    fps = 1 / (currentT - previousT)
    previousT = currentT

    cv2.putText(frame, 'FPS:' + str(int(fps)), (15, 700), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                color=(255, 255, 255), thickness=5)
    cv2.putText(frame, 'FPS:' + str(int(fps)), (15, 700), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                color=(98, 98, 98), thickness=2)

    cv2.namedWindow('paint', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('paint', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('paint', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
