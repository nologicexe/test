import cv2

cascPath=r'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

class face_recogn(object):
      def __init__(self,frame):
         self.image=frame     
      def face_recogn(self):
          try:
              gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
              faces = faceCascade.detectMultiScale(gray,
                                              scaleFactor=1.1,
                                              minNeighbors=5,
                                              minSize=(30,30),
                                              flags=cv2.CASCADE_SCALE_IMAGE)
          except cv2.error:
              faces=[]

#          print('face count is ',len(faces))
#          if len(faces)>0:
#             print('face/faces detected')
#             command='start'
#          else:
#             command='stop'
          return faces

      def draw_face(self,faces,frame):
          for (x, y, w, h) in faces:
              cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
          cv2.imshow('face', frame)
      
      def compare_faces(self,face0,face1):
          for (x0, y0, w0, h0) in face0:
              for (x1, y1, w1, h1) in face1:
                  if x0==x1 and y0==y1 and w0==w1 and h0==h1:
                     cpr=True
                     print('faces are identical.')
                  else:
                     cpr=False
                     print('faces are not identical.')
          return cpr
          
#if __name__ == '__main__':
#    #video_ID="rtsp://admin:Qw123456@94.25.153.216:554/LiveMedia/ch1/Media1"
#    video_ID=0
#    cap=cv2.VideoCapture(video_ID)
#    while(cap.isOpened()):
#          ret, frame = cap.read()
#          if ret==True:
#             fc=face_recogn(frame)
#             faces=fc.face_recogn()[1]
#             fc.draw_face(faces,frame)
#             #cv2.imshow(str(video_ID),frame)
#          if 0xFF == ord('q'):#если нажать q запись прервется
#             print(int(timing))
#             break
#    cap.release()
#    cv2.destroyAllWindows()
