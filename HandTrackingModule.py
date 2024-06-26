import mediapipe as mp
import time
import cv2
import math
import numpy as np
class HandDetector():
    def __init__(self, mode=False, maxHands=2, complexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
    
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if(draw):
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img
    
    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        xList = []
        yList = []
        bbox = None
        if self.results.multi_hand_landmarks:
            #myHand = self.results.multi_hand_landmarks[handNo]
            for handLms in self.results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    xList.append(cx)
                    yList.append(cy)
                    lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (0,255,0), cv2.FILLED)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin - 10, ymin- 10, xmax + 10, ymax + 10
        return lmList, bbox

    def fingersUp(self, lmList):
        finger_indexes = [(20, 18), (16, 14), (12, 10), (8,6)]
        finger_states = [0] * 5
        
        for i, (finger_idx, joint_idx) in enumerate(finger_indexes):
            if lmList[finger_idx][2] < lmList[joint_idx][2]:   
                finger_states[i] = 1
            elif lmList[finger_idx][2] > lmList[joint_idx][2]:
                finger_states[i] = 0
            
        finger_states[4] = 1 if lmList[4][1] > lmList[3][1] else 0    
        
        return finger_states
        # pinky, ring, middle, index, thumb
    
    
    def findDistance(self, lmList, finger1 = 4, finger2 = 8, img = None, draw=False):
        x1, y1 = lmList[finger1][1], lmList[finger1][2]
        x2, y2 = lmList[finger2][1], lmList[finger2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        
        if draw:
            cv2.circle(img, (x1, y1), 10, (0, 255,0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (0, 255,0), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (0,255, 0), 2, cv2.LINE_AA)
        
        return length, img, [x1, y1, x2, y2, cx, cy]

def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    cTime = 0
    detector = HandDetector()
    while True:
        success, image = cap.read()
        
        img = detector.findHands(image)
        lmList = detector.findPosition(img)
        
        if len(lmList) > 0:
            print(lmList)
        
        cTime = time.time()        
        fps = 1/(cTime - pTime)        
        pTime = cTime
         
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)        
                
        cv2.imshow("Image", img)
        cv2.waitKey(1)
    
if __name__ == '__main__':
    main()
