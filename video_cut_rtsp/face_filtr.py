import cv2
import os
import time
from face_recogn import face_recogn
#import main
import glob
import shutil

for p in glob.glob('video/*.avi'):
    print(p)
    cap=cv2.VideoCapture(p)
    f_index=0
    fr=[]
    j=0
    fps = round(float(cap.get(cv2.CAP_PROP_FPS)), 2)
    frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))#frame count is
    length = round(frame / fps, 2)
    time_start=time.process_time()
    print('time video is ',length)
    timing = round(time.process_time() - time_start, 0)
    while cap.isOpened()==True and timing<length:
         timing = round(time.process_time() - time_start, 0)
         print(timing)
         ret, frame = cap.read()
         fr.append(frame)
#         print(len(fr))
         if len(fr)==3 and j!=0:
#           print('clear cash...')
            fr.clear()
         if ret==True and len(fr)==2:
            fc0=face_recogn(fr[0]).face_recogn()
            fc1=face_recogn(fr[1]).face_recogn()
            if len(fc0)>0 and len(fc1)>0: 
               cpr=face_recogn(fr[1]).compare_faces(fc0,fc1)
            else:
               cpr=True
            print('static check: ',cpr)

            if len(fc0)>0 and len(fc1)>0 and cpr==False:
               print('faces found...')
               f_index=+1
         j+=1
             
#            else:
#               print('file will to delete...')
#               os.remove(p)
    cap.release()
    
    if f_index>0:
       shutil.move(p,'video_new')  
    