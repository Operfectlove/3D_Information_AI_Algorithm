#스테레오 이미지로 3D 좌표값 위치 추정(에듀테크온 2022.11.9)
import numpy as np
import cv2
from matplotlib import pyplot as plt
imgL = cv2.imread('tsukuba_l.png',0) #스테레오 이미지
imgR = cv2.imread('tsukuba_r.png',0)
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,'gray')
plt.show()
#ㅇㅇ