

import cv2
import mediapipe as mp
import HandTrackingModule as htm
import numpy as np
import time
import os
wCam, hCam = 640, 480

cTime = 0
pTime = 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = "handImages"
myList = os.listdir(folderPath)
overlayList = []

for imgPath in myList:
    image = cv2.imread(f"{folderPath}/{imgPath}")
    overlayList.append(image)

detector = htm.HandDetector(maxHands=1)
fingers = [0] * 5
text_size = cv2.getTextSize(f"0 0 0 0 0", cv2.FONT_HERSHEY_PLAIN, 2, 2)
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    
    lmList, bbox = detector.findPosition(img, draw=False)
    
        
    if len(lmList) > 0:
        detector.fingersUp(lmList)
        fingers = detector.fingersUp(lmList)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        print(x1, y1, x2, y2)
    
    h, w, c = overlayList[0].shape
    img[50:50+h, 50:50+w] = overlayList[0]
    
    
    
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    
    
  
    img_h, img_w, img_c = img.shape
    
    cv2.putText(img, str(f"{fingers[0]} {fingers[1]} {fingers[2]} {fingers[3]} {fingers[4]}"), (img_w // 2 - text_size[0][0] // 2, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    
    cv2.putText(img, str(int(fps)), (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
    
    cv2.imshow('img', img)
    cv2.waitKey(1)