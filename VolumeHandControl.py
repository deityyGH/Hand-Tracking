

import cv2
import mediapipe as mp
import HandTrackingModule as htm
import numpy as np
import time
import math

### pycaw
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

wCam, hCam = 640, 480

cTime = 0
pTime = 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.HandDetector(maxHands=1, detectionCon=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

masterVol = volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()

minVol = volRange[0]
maxVol = volRange[1]

minRange = 20
maxRange = 150

vol = 0
volBar = np.interp(masterVol, [-65, 0], [400, 150])
volPercent = np.interp(masterVol, [-65, 0], [0, 100])

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=False)
    
    if len(lmList) > 0:
        length, img, coords = detector.findDistance(lmList, img=img, draw=True)
        
        cv2.circle(img, (coords[0], coords[1]), 10, (0, 255,0), cv2.FILLED)
        cv2.circle(img, (coords[2], coords[3]), 10, (0, 255,0), cv2.FILLED)
        cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), (0,255, 0), 3, cv2.LINE_AA)
        
        #range 20 - 180
        vol = np.interp(length, [minRange, maxRange], [minVol, maxVol])
        volBar = np.interp(length, [minRange, maxRange], [400, 150])
        volPercent = np.interp(length, [minRange, maxRange], [0, 100])
        #print(vol)
        
        if length < minRange or length > maxRange:
            cv2.circle(img, (coords[4],coords[5]), 10, (255, 0, 0), cv2.FILLED)
        else:
           cv2.circle(img, (coords[4],coords[5]), 10, (0, 255, 0), cv2.FILLED) 
        
        volume.SetMasterVolumeLevel(vol, None)
        
    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
    
    cv2.putText(img, str(int(volPercent)), (40, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    
   
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    
    cv2.imshow('img', img)
    cv2.waitKey(1)