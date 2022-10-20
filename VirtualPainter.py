import cv2
import time
import HandTrackingModule as htm
import numpy as np
import os

brushThickness = 15
earaserThickness = 175

# Import header images
folderPath = "Header"
myList = os.listdir(folderPath)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
header = overlayList[4]
drawColor = (226, 182, 122)


# initialize camera
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionConfidence=0.8)
xPrev, yPrev = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)
infoMode = True

while True:

    # 1. Import image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # 2. Find Hand Landmarks
    img = detector.findHands(img)
    landmarkList = detector.findPosition(img, draw=False)
    # If hand detected
    if len(landmarkList) != 0:
        # tip of index and middle fingers
        x1, y1 = landmarkList[8][1:]
        x2, y2 = landmarkList[12][1:]

        # 3. Check which fingers are up
        fingers = detector.fingersUp()

        # 4. If Selection mode -At least two fingers are up
        if not(fingers[1] or fingers[2] or fingers[3] or fingers[4]):
            xPrev, yPrev = 0, 0
            # Checking for the click
            if y1 < 125:
                # Help menu
                if 40 < x1 < 240:
                    infoMode = True
                    header = overlayList[4]

                else:
                    infoMode = False
                    # Blue
                    if 290 < x1 < 490:
                        header = overlayList[0]
                        drawColor = (226, 182, 122)  # BGR
                    # Red
                    elif 540 < x1 < 740:
                        header = overlayList[1]
                        drawColor = (116, 115, 239)  # BGR
                    # Green
                    elif 790 < x1 < 990:
                        header = overlayList[2]
                        drawColor = (130, 215, 180)  # BGR
                    # Earaser
                    elif 1040 < x1 < 1240:
                        header = overlayList[3]
                        drawColor = (0, 0, 0)
            if not(infoMode):
                cv2.rectangle(img, (x1, y1-25), (x2, y2+25),
                              drawColor, cv2.FILLED)

        # 5. If Drawing mode -Index finger is up
        if not(infoMode) and fingers[1] and fingers[2] and fingers[3] and fingers[4]:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)

            if xPrev == 0 and yPrev == 0:
                xPrev, yPrev = x1, y1

            # Erase
            if drawColor == (0, 0, 0):
                cv2.line(img, (xPrev, yPrev), (x1, y1),
                         drawColor, earaserThickness)
                cv2.line(imgCanvas, (xPrev, yPrev), (x1, y1),
                         drawColor, earaserThickness)
            # Draw
            else:
                cv2.line(img, (xPrev, yPrev), (x1, y1),
                         drawColor, brushThickness)
                cv2.line(imgCanvas, (xPrev, yPrev),
                         (x1, y1), drawColor, brushThickness)

            xPrev, yPrev = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # Setting the header image
    img[0:125, 0:1280] = header
    if infoMode:
        img[124:720, 0:310] = overlayList[5]
    cv2.imshow("Image", img)
    # cv2.imshow("Canvas", imgCanvas)
    # cv2.imshow("Inv", imgInv)
    cv2.waitKey(1)
    if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
        break
