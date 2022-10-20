import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionConfidence=0.5, trackConfidence=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConfidence = detectionConfidence
        self.trackConfidence = trackConfidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionConfidence, self.trackConfidence)
        self.mpDraw = mp.solutions.drawing_utils

        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLandmarks in self.results.multi_hand_landmarks:
                if draw:                
                    #display the landmarks of the hands 
                    self.mpDraw.draw_landmarks(img, handLandmarks, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=False):

        self.landmarkList = []

        if self.results.multi_hand_landmarks:
            #Get hand number handNo
            myHand = self.results.multi_hand_landmarks[handNo]

            #coordinates for each landmark
            for id, lm in enumerate(myHand.landmark) :
                height, width, channels = img.shape
                centerX, centerY = int(lm.x * width), int(lm.y * height)
                # print(id, centerX, centerY)
                self.landmarkList.append([id, centerX, centerY])
                if draw:
                    cv2.circle(img, (centerX, centerY), 15, (255,0,255), cv2.FILLED)
        
        return self.landmarkList

    def fingersUp(self):
        fingers = []

        # Thumb
        if self.landmarkList[ self.tipIds[0] ][1] > self.landmarkList[ self.tipIds[0]-1 ][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        
        # 4 Fingers
        for id in range(1,5):
            if self.landmarkList[ self.tipIds[id] ][2] < self.landmarkList[ self.tipIds[id]-2 ][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers


def main():
    prevTime = 0
    currTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector() 

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        landmarkList = detector.findPosition(img)
        if len(landmarkList) != 0:
            print(landmarkList[4])

        #Frame rate
        currTime = time.time()
        fps = 1/(currTime - prevTime)
        prevTime = currTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()