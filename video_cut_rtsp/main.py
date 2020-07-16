# -*- coding: utf-8 -*-
import time
import cv2
from scsissors_video import video_scissors
import os

def main(video_ID):
    print('started')
	#video_ID="rtsp://admin:123456Z@94.25.153.216:80/ch0_0.264"
    #video_ID="rtsp://admin:123456Z@94.25.153.216:80/ch0_0.264"
    #video_ID=0
    #video_ID="rtsp://admin:Qw123456@94.25.153.216:554/LiveMedia/ch1/Media1"
    
    cap=cv2.VideoCapture(video_ID)
    fps = round(float(cap.get(cv2.CAP_PROP_FPS)), 2)
    frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))#frame count is
    t = frame * 25 / 60
    length = round(frame / fps, 2)
    print('time is ',t)
    print('length is ', length)
  
    time_video=5#время ролика
    time_start=time.process_time()
    time_end=length#время записи
    num=int(time_end/time_video)#число роликов
    #path=r"C:\Users\StudyLie2\Desktop\video_cutting\video_cut"
    #path=r"/home/web/project_video_cap"
    path=r"/home/py/video_cut"
    file_path_video=path+r"/video"
    file_path_json=path+r"/json"

    time_start=time.process_time()
    timing = round(time.process_time() - time_start, 0)
    video=video_scissors(cap,video_ID)
    video.scissors(num,time_start,time_video,time_end,file_path_video,file_path_json)

main('test.mp4')
