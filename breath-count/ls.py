import cv2
import numpy as np
import math
import pandas as pd
import csv
import matplotlib.pyplot as plt 
import imutils
'''
def sort_col(i):
    return i[n]
def frame_diff(prev_frame, cur_frame, next_frame):
	diff_frames1 = cv2.absdiff(next_frame, cur_frame)
	diff_frames2 = cv2.absdiff(cur_frame, prev_frame)
	return cv2.bitwise_and(diff_frames1, diff_frames2) +250 #достаточная чувствительность для дыхания

def get_frame(cap):
	ret, frame = cap.read()
	frame = cv2.resize(frame, None, fx=scaling_factor,
		fy=scaling_factor, interpolation=cv2.INTER_AREA)	
	return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
	'''
x2=[]
rast=[]
sqr=[]
z1=[]
squ=[]
yg=[]
left1=[]
repl=[]
z=[]
i=0
cap = cv2.VideoCapture('test6.mov') 
frameRate=cap.get(5)/3 # cap.get(5)- возвращает количество кадров в секунду
count=0
while(cap.isOpened()): 
	frameId=cap.get(1)
	ret, frame = cap.read() 
	if (ret != True):
			break
	if (frameId % math.floor(frameRate)==0):	#берем каждый 3й кадр 

	# находим лица на этих кадрах 
		face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml') 
		faces = face_cascade.detectMultiScale(frame, scaleFactor=1.01, minNeighbors= 5,
			minSize=(20, 20), flags=cv2.CASCADE_SCALE_IMAGE)    

		for (x, y, w,h) in faces:
			# берем координаты лица(квадрат) и обрезаем по нему вверх и вниз,
			# захватывая плечи
			frame = frame[y+h:y+h*2,x-int(2*w):x+int(2.3*w)]
			gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) # делаем изображение серым
			kernel_size = 5
			blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
			low_threshold = 30 # чем меньше тем больше контуров находит, но может находить
			# и всякий мусор, если сделать очень маленьким
			high_threshold = 150
			edges = cv2.Canny(blur_gray, low_threshold, high_threshold) # находим контуры
			if edges is not None: # проверяем на присутствие контуров на изображение
				#cv2.imshow("Image", edges)
				#cv2.waitKey(1000)
				cnts = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
				cnts = imutils.grab_contours(cnts)
				for c in cnts:
					x2.append(len(c))
					x2=sorted(x2)
				# записываем все контуры в массив по нарастанию их длины
				for c in cnts:
					if len(c)== x2[len(x2)-1]:
						left = c
					elif len(c)== x2[len(x2)-2]:
						right = c
				# записываем координаты двух самых длинных контуров
				# левой и правой части одежды соответсвенно		
				left=sorted(left,key=lambda x:x[0][1])	
				right=sorted(right,key=lambda x:x[0][1])
				unil = np.unique(left)
				unir = np.unique(right)
				l=0
				r=0
				k=0
				left1=[]
				maximum=[]
				# находим все горизонтальные линии(в которых координата y - одинаковая)
				# и выбираем из них максимальное значение x
				while l<len(left)-1: # для 1го контура
					if left[l][0][1]==left[l+1][0][1]:
						k+=1
					else:
						if k>=1:
							maximum = max(left[l-k:l][0])
							left1.append(maximum)
						else:	
							left1.append(left[l][0])
						k=0	
					l+=1
				right1=[]	
				maximum=[]
				k=0
				while r<len(right)-1: # для второго
					if right[r][0][1]==right[r+1][0][1]:
						k+=1
					else:
						if k>=1:
							maximum = max(right[r-k:r][0])
							right1.append(maximum)
						else:	
							right1.append(right[r][0])
						k=0
					r+=1		
				# пробегаем по всем координатам и ищем совпадение по y,
				# высчитываем расстояние между точками левого и правого контура
				# и записываем их в массив
				for l in left1:
					for r in right1:
						if l[1]==r[1]:
							ras=abs(l[0]-r[0])
							rast.append(ras)
				# суммируем все расстояния - это и есть площадь человека в кадре			
				sq=sum(rast) 
				rast=[]	
				z.append(sq) # записываем в массив все площади на всех кадрах	
				x2=[]		
cap.release()
print(len(z))
count=0
i=0
zel=[]
av=max(z)*0.3
# проверяем на выбросы
while i< len(z):
	if z[i]>av :
		zel.append(z[i])
		count+=1
	i+=1	

plt.plot(zel,color = 'red', marker = "o")
plt.show() 
a=[]
h=[]
i=0
while i<count:
	a.append(zel[i])
	i+=1
	h.append(i)
# записываем в таблицу площадь и номер по возрастанию на случай дальнейшей работы
# с этими данными
a=np.array(a)
h=np.array(h)
d=np.column_stack((a, h))
df=pd.DataFrame(d)
df=np.array(df)
b=open('frame1.csv','w')
c=csv.writer(b)
c.writerows(df)
b.close()
