### index finger -> move mouse
### ring finger to thumb -> double click 
### middle finger to thumb -> single click
import cv2
import mediapipe as mp
import HandTrackingModule as htm
import autopy
import numpy as np
import time
import os
import pyautogui as pag
wCam, hCam = 640, 480
wScreen, hScreen = pag.size()
frameReduction = 130

cTime = 0
pTime = 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
finger = [0] * 5
detector = htm.HandDetector(maxHands=2)

clickRange = 30
smoothening = 5
plocX, plocY = 0,0
clocX, clocY = 0,0

clicks = 0
clicked = False
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=False)
    cv2.rectangle(img, (frameReduction, frameReduction), (wCam - frameReduction, hCam - frameReduction), (255, 0, 255), 2)
    if len(lmList) > 0:
        fingers = detector.fingersUp(lmList)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.circle(img, (int(lmList[8][1]), int(lmList[8][2])), 5, (255, 0, 255), 2)
        
        x1, y1 = lmList[8][1:]
        x3 = np.interp(x1, (frameReduction, wCam - frameReduction), (0, wScreen))
        y3 = np.interp(y1, (frameReduction, hCam - frameReduction), (0, hScreen))

        if x3 > 0 and x3 < wScreen and y3 > 0 and y3 < hScreen:
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening
            pag.moveTo(wScreen - x3, y3)
            plocX, plocY = clocX, clocY
            length_single, img, coords_single = detector.findDistance(lmList, 4, 12, img, True)
            length_double, img, coords_double = detector.findDistance(lmList, 4, 16, img, True)
            cv2.circle(img, (coords_single[4],coords_single[5]), 10, (0, 255, 0), cv2.FILLED)
            if length_single < clickRange:
                cv2.circle(img, (coords_single[4],coords_single[5]), 10, (255, 0, 0), cv2.FILLED)
                pag.click()
                time.sleep(0.5) 
            
            if length_double < clickRange:
                cv2.circle(img, (coords_double[4],coords_double[5]), 10, (255, 0, 0), cv2.FILLED)
                pag.click()
                time.sleep(0.1) 
                pag.click()

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    img_h, img_w, img_c = img.shape
    
    cv2.putText(img, str(int(fps)), (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
    
    cv2.imshow('img', img)
    cv2.waitKey(1)