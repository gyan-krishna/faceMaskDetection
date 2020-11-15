# -*- coding: utf-8 -*-
"""
----------Phenix Labs----------
Created on Sun Nov 15 13:05:09 2020
@author: Gyan Krishna
Topic: Face Mask Detection
"""

import tensorflow as tf
from tensorflow import keras

model =  keras.models.load_model('maskDetectorModel')
image_size = (180, 180)

import cv2 
import numpy as np 
from PIL import Image

vid = cv2.VideoCapture(0)
ret, frame = vid.read()
y,x,_ = frame.shape

outline = 255 - cv2.imread(r"images/outline.png")
outline = cv2.resize(outline, (x,y))

while(True): 
    ret, frame = vid.read()
    
    x1, y1 = (int(x/2- x/6),int(y/20))
    x2, y2 = (int(x/2+x/6),int(y-y/4) )
    
    crop = frame[y1:y2, x1:x2]
    
    img = Image.fromarray(crop, 'RGB')
    img = cv2.resize(frame, image_size)
    img_arr = np.array(img)
    img_arr = np.expand_dims(img_arr, 0)
    
    predictions = model.predict(img_arr)
    score = predictions[0]
    
    if (1-score) > 0.85:
        txt = "mask"
        color = (0, 255, 0)
    else:
        txt = "no mask"
        color = (0, 0, 255)
        
    
    #txt =  "%.2f percent mask"% (100 * (1 - score))
    frame = cv2.putText(frame, txt, (50,100), cv2.FONT_HERSHEY_SIMPLEX,  1, color, 2, cv2.LINE_AA) 
    frame = cv2.add(frame, outline)  
    frame = cv2.putText(frame, "Allign face to the outline", ( int(x/5), int(y - y/13)), cv2.FONT_HERSHEY_SIMPLEX,  1, (0,0,255), 2, cv2.LINE_AA)
    
    
    cv2.imshow("frame",frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    
vid.release() 
cv2.destroyAllWindows()