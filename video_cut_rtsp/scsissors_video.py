import cv2
import time
import json
from datetime import datetime
import random
import os
#from parser import JsonToDb

class video_scissors(object):
      def __init__(self,cap,videoid):
          self.videoid=videoid
          self.cap=cap
      
      def video_record(self,file_path,time_video,name):
          fps = round(float(self.cap.get(cv2.CAP_PROP_FPS)), 2)
          print('fps=',fps)
          frame_width = int(self.cap.get(3))
          frame_height = int(self.cap.get(4))
          pathvideo=file_path+str(name)
          pathvideo=os.path.abspath(pathvideo)
          out = cv2.VideoWriter(pathvideo,
                                cv2.VideoWriter_fourcc('M','J','P','G'), 
                                fps,#fps
                                (frame_width,frame_height))
          time_start=time.process_time()
          while(self.cap.isOpened()):
               timing = time.process_time() - time_start
               ret, frame = self.cap.read()
               #cv2.imshow(str(self.videoid),frame)
               if ret==True:
                  out.write(frame)
               if int(timing) ==int(time_video) or 0xFF == ord('q'):#если нажать q запись прервется
                  print(int(timing))
                  break 
          out.release()
          cv2.destroyAllWindows()
      
      def scissors(self,n_video,time_start,time_video,time_end,file_path_video,file_path_json):
          i=0
          #time_start=time.process_time()
          timing = round(time.process_time() - time_start, 0)
          while(i<n_video):
                if timing<time_end:
                   timing = round(time.process_time() - time_start, 0)
                   time_now=datetime.strftime(datetime.now(), "%Y.%m.%d_%H-%M-%S")
                   tm=str(time_now)
                   #tm=int(random.random()*10000)
                   name=r"/video_"+str(i)+'_'+str(tm)+'.avi'
                   link=file_path_video+name
                   link=os.path.abspath(link)
                   self.video_record(file_path_video,time_video,name)
                   d={
                       "camera_personal_id": "7ffe5832-a16f-4bee-ad87-1824fc7e5d5e",
                       "camera_adress": "rtsp:something-something",
                       "video_name": name,
                       "file_url": link
                     }
                   json_var = json.dumps(d)
                   #JsonToDb(json=json_var)
                   path_json=file_path_json+r"/json_"+str(i)+'_'+str(tm)+r"_id_videos.json"
                   path_json=os.path.abspath(path_json)
                   with open(path_json, "w") as write_file:
                       json.dump(d, write_file)

                   with open(path_json, "r") as read_file:
                       data = json.load(read_file)
                       print(data)
                   i+=1

#coments: if error "Could not open codec 'libopenh264': Unspecified error" occurs 
#then to https://stackoverflow.com/questions/41972503/could-not-open-codec-libopenh264-unspecified-error
