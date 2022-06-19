# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 21:13:09 2021

@author: peter_ex
"""
###########臉眼辨識Dataset產生(Yolov5專用)###########
import cv2
import numpy as np
from tqdm import tqdm, trange
import os
from os import listdir
from os.path import isfile, isdir, join

######只須設定這裡######
mypath_img = "I:\\facedata\\test" #圖片資料夾
Extension = 'png'#圖片副檔名(限一種，無法複數檔名)
#####################

files = listdir(mypath_img)
mypath_img_out = mypath_img+'\\images' #輸出image資料夾(可自行改名)
os.mkdir(mypath_img_out)
mypath_txt = mypath_img+'\\labels' #輸出label資料夾(可自行改名)
os.mkdir(mypath_txt)

#載入臉與眼的特徵數據(需與python檔同路徑)    
face_xml = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_xml = cv2.CascadeClassifier('haarcascade_eye.xml')
for  f in tqdm(files):
    fullpath = join(mypath_img, f)

    #讀取圖片
    img = cv2.imread(fullpath)
    #圖片長寬
    size = img.shape
    size_w = size[1]
    size_h = size[0]
    #灰階
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 1.灰色影象 2.縮放係數 3.目標大小
    faces = face_xml.detectMultiScale(gray, 1.3, 5)
    
    #print('face = ',len(faces))
    #輸出人臉座標: 分類 x y 寬 高
    faceData = ''
    for (x,y,w,h) in faces:
        #cv2.rectangle(img, (x,y), (x + w, y + h), (255,0,0), 2)
        roi_face = gray[y:y+h,x:x+w]
        #roi_color = img[y:y+h,x:x+w]
        x1 = (x+(w/2))/size_w
        y1 = (y+(h/2))/size_h
        w1 = w/size_w
        h1 = h/size_h
        faceData = '0 '+str(x1)+' '+str(y1)+' '+str(w1)+' '+str(h1)+'\n'
        eyes = eye_xml.detectMultiScale(roi_face)
        #print('eyes = ',len(eyes))
        for (ex,ey,ew,eh) in eyes:
            #cv2.rectangle(roi_color, (ex,ey),(ex + ew, ey + eh), (0,255,0), 2)
            ex1 = ((ex+x)+(ew/2))/size_w
            ey1 = ((ey+y)+(eh/2))/size_h
            ew1 = ew/size_w
            eh1 = eh/size_h
            faceData =faceData+ '1 '+str(ex1)+' '+str(ey1)+' '+str(ew1)+' '+str(eh1)+'\n'
    if faceData!='' :
        #print(faceData)
        f1 = f.replace('.'+Extension,'')
        txt = open(mypath_txt+'\\'+f1+'.txt', 'w')
        txt.write(faceData)
        txt.close()
        img_out = cv2.imread(fullpath)
        cv2.imwrite(mypath_img_out+'\\'+f, img_out)

