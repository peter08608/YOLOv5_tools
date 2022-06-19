# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 22:20:41 2021

@author: peter_ex
"""

###########Dataset產生train valid test(Yolov5專用)###########
import cv2
from tqdm import tqdm, trange
import os
from os import listdir
from os.path import isfile, isdir, join
import time
import shutil

######只須設定這裡######
newfilename = 'detect_data_separate' #分離後資料夾名子
mypath_img = r"C:\Users\peter\Desktop\yolov5\detect_data" #圖片及標籤所處路徑(資料夾名稱需下面更改)
files = listdir(mypath_img)
mypath_img_out = mypath_img+'\\image' #輸出image資料夾(可自行改名)
mypath_txt = mypath_img+'\\label' #輸出label資料夾(可自行改名,限txt)
proportion = '5:1:0' #輸出比例{train:valid:test}，比例需 大:中:小 ，且test可為0
label = 'txt' #label副檔名
#####################

#資料夾初始化
print('Create new folder...')
proportion = proportion.split(':')

newfilename = mypath_img+'\\'+newfilename
os.mkdir(newfilename)
time.sleep(1)

trainfile = newfilename+'\\trains'
os.mkdir(trainfile)
time.sleep(1)
trainimg = trainfile+'\\images'
os.mkdir(trainimg)
time.sleep(1)
traintxt = trainfile+'\\labels'
os.mkdir(traintxt)
time.sleep(1)

validfile = newfilename+'\\valid'
os.mkdir(validfile)
time.sleep(1)
validimg = validfile+'\\images'
os.mkdir(validimg)
time.sleep(1)
validtxt = validfile+'\\labels'
os.mkdir(validtxt)
time.sleep(1)

if int(proportion[2])!= 0:
    testfile = newfilename+'\\test'
    os.mkdir(testfile)
    time.sleep(1)
    testimg = testfile+'\\images'
    os.mkdir(testimg)
    time.sleep(1)
    testtxt = testfile+'\\labels'
    os.mkdir(testtxt)
    time.sleep(1)
print('Done!')

files = listdir(mypath_img_out)
total_step = int(proportion[0])+int(proportion[1])+int(proportion[2])
step = 0
print('Separating...')
time.sleep(1)

for f in tqdm(files):
   step += 1
   f1 = f.split('.')
   # print(step,':',f)
   if int(proportion[2])>=step:
       shutil.copyfile(mypath_img_out+'\\'+f, testimg+'\\'+f)
       shutil.copyfile(mypath_txt+'\\'+f1[0]+'.'+label, testtxt+'\\'+f1[0]+'.'+label)
   elif int(proportion[2])<step and int(proportion[1])>=step:
       shutil.copyfile(mypath_img_out+'\\'+f, validimg+'\\'+f)
       shutil.copyfile(mypath_txt+'\\'+f1[0]+'.'+label, validtxt+'\\'+f1[0]+'.'+label)
   else:
       shutil.copyfile(mypath_img_out+'\\'+f, trainimg+'\\'+f)
       shutil.copyfile(mypath_txt+'\\'+f1[0]+'.'+label, traintxt+'\\'+f1[0]+'.'+label)
       if step == total_step:
           step = 0



