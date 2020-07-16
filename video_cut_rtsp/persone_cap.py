import cv2
import numpy as np
import pandas as pd
import json
import os
from face_recogn import face_recogn
import glob

def border_up_left(x,y,w,h):
    if x<=0:
        x=0
    if y<=0:
        y=0
    return x,y
def border_down_right(x,y,w,h):
    if x>w:
        x=w
    if y>h:
        y=h
    return x,y

def crop_persone(faces,frame,i,path):
#    img = cv2.imread(frame)
    height, width = frame.shape[:2]
#    j=0
#    for face in faces:
    for (x, y, w, h) in faces:
        x0,y0=border_up_left(x-int(1.6*w),y-int(0.5*h),width,height)

        xend,yend=border_down_right(x + int(2.6*w),y + int(10*h),width,height)

#        frame=cv2.rectangle(frame, (x0,y0),(xend,yend), (0, 0, 255), 2)
        
        crop_img = frame[y0:yend, x0:xend]
#        cv2.imshow("cropped", crop_img)
        cv2.imwrite(path+'\crop_frame_num_'+str(i)+'face_num_'+str(len(faces))+'.jpg', crop_img)
#        cv2.imwrite(path+'\frame_num_'+str(i)+'face_num_'+str(len(faces))+'.jpg', frame)
#        j+=1

def save_cadr(dir):
    #video_name, __ = os.path.splitext(dir)
    video_name, ext = os.path.splitext(os.path.basename(dir))
    #video_name=dir.split("\")[0]
    print('video name is ',video_name)
    print('directory is ',dir)
    path=r"C:\Users\StudyLie2\Desktop\video_cutting\video_cut\images"
    path=os.path.join(path, str(video_name))
    try:
       os.mkdir(path)
    except:
       pass
    cap = cv2.VideoCapture(dir)
    i=0
    ret=True
    fc=[]
    while cap.isOpened()==True and ret==True:
          ret, frame = cap.read()
          #faces=face_recogn(frame).face_recogn()
          
          if i%2==0 and i!=0:    
             #print('faces count is ',len(faces))
             print(path)
             fc.append(frame)
             if len(fc)==3:
                fc.clear()
             
             if len(fc)==2:
                fc0=face_recogn(fc[0]).face_recogn()
                fc1=face_recogn(fc[1]).face_recogn()
                if len(fc0)>0 and len(fc1)>0: 
                   cpr=face_recogn(fc[1]).compare_faces(fc0,fc1)
                   print('faces found...')
                   crop_persone(fc1,frame,i,path)
          i+=1

path=r"C:\Users\StudyLie2\Desktop\video_cutting\video_cut\video_new"
for p in glob.glob(path+'/*.avi'):
    print(p)
    save_cadr(p)