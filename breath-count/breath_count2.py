from functools import reduce
import cv2
import numpy as np
import math
#import pandas as pd
#import csv
import matplotlib.pyplot as plt
import imutils
from scipy.signal import argrelmax

# из массива к которому применяют эту функцию делается массив с уникальными координатами contour_unique
#массив 2мерный
def compare_reduce(x,y,conotur_unique):
	if x[0][1]==y[0][1]:
		x[0][0]=max(x[0][0],y[0][0])
	else:
		conotur_unique.append(x)
		x=y 
	return x		

def distance_reduce(x,y):
	if x[0][1]==y[0][1]:
		distance=abs(x[0][0]-y[0][0])
	else:
		distance=0	
	return distance	


# Low-Pass Filter (Fast Fourier Transform)
def FFT():			
	fc=0.1 # частота дискретизации
	b = 0.1 # полоса перехода
	N = int(np.ceil((4 / b)))
	if not N % 2: N += 1
	n = np.arange(N)
	sinc_func = np.sinc(2 * fc * (n - (N - 1) / 2.))
	window = 0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + 0.08 * np.cos(4 * np.pi * n / (N - 1))
	sinc_func = sinc_func * window
	sinc_func = sinc_func / np.sum(sinc_func)
	return sinc_func


length_contour=[]
distance_array=[]
square_array=[]
kernel_size = 5
low_threshold = 20 # чем меньше тем больше контуров находит
high_threshold = 150
cap = cv2.VideoCapture('test6.mov') 
frameRate=cap.get(5)/3 # cap.get(5)- возвращает количество кадров в секунду, делим на 3 -
# эмперически самое нормальное соотношение производительности и точности
count=0
while(cap.isOpened()): 
	frameId=cap.get(1)
	ret, frame = cap.read() 
	if (ret != True):
			break
	if (frameId % math.floor(frameRate)==0):	#берем каждый 3й кадр
												# находим лица на этих кадрах 
		face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml') 
		faces = face_cascade.detectMultiScale(frame, scaleFactor=1.01, minNeighbors=5, 
			minSize=(20, 20), flags=cv2.CASCADE_SCALE_IMAGE)    
		for (x, y, w, h) in faces:
			# берем координаты лица(квадрат) и обрезаем по нему вверх и вниз, захватывая плечи
			below_face = y+h
			below_shoulder=y+h*2
			left_from_face=x-int(2.3*w)
			right_from_face=x+int(2.3*w)
			frame = frame[below_face:below_shoulder, left_from_face:right_from_face]
			gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # делаем изображение серым
			blur_gray_img = cv2.GaussianBlur(gray_img, (kernel_size, kernel_size), 0)
			edges = cv2.Canny(blur_gray_img, low_threshold, high_threshold) # находим контуры
			if edges is None: # проверяем на присутствие контуров на изображение
				break
				
			contours = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
			contours = imutils.grab_contours(contours)
			length_contour=list(map(lambda x: len(x), contours)) #записываем длины контуров
			length_contour=sorted(length_contour, reverse=True) # сортируем длины по убыванию
			for c in contours:# записываем координаты двух самых длинных контуров - левой и правой части одежды соответсвенно		
				if len(c)== length_contour[0]: left_contour = c
				elif len(c)== length_contour[1]: right_contour = c
			#сортируем их по возрастанию
			left_contour=sorted(left_contour, key=lambda x: x[0][1])
			right_contour=sorted(right_contour, key=lambda x: x[0][1])	
			#находим уникальные координаты и записываем их в _unique массивы
			left_contour_unique=[]
			product = reduce((lambda x, y: compare_reduce(x,y,left_contour_unique)), left_contour)
			right_contour_unique=[]
			product = reduce((lambda x, y: compare_reduce(x,y,right_contour_unique)), right_contour)
			# пробегаем по всем координатам обеих контуров и ищем совпадение по y
			for l in left_contour_unique:
				for r in right_contour_unique:
					distance1=distance_reduce(l,r)      # высчитываем расстояние между точками левого и правого контура
					distance_array.append(distance1) # записываем их в массив
			# суммируем все расстояния - это и есть площадь человека в кадре		
			square=sum(distance_array) 
			square_array.append(square) # записываем в массив все площади на всех кадрах	
			#обнуляем вспомогательные массивы 
			distance_array=[]	
			length_contour=[]	
cap.release()
# проверяем на выбросы
minimum_square_filter=0.5*sum(square_array)/len(square_array)
square_array_average=list(filter(lambda x: x>minimum_square_filter, square_array))
new_signal = np.convolve(square_array_average, FFT()) # делаем свертку Фурье преобр.
new_signal=list(filter(lambda x: x>1000, new_signal)) # обрезаем левый и правый кусок sinc_func
new_signal=np.array(new_signal)
breath_count=argrelmax(new_signal)
print('количество вдохов',len(breath_count[0]))
plt.plot(new_signal,color = 'red', marker = "o")	
plt.show()


'''
Запись в таблицу данных о координатах и кадрах
row_1=[]
row_1.append('Square')
row_2=[]
row_2.append('Frame')
i=0
while i<count:
	row_1.append(square_array_average[i])
	i+=1
	row_2.append(i)
# записываем в таблицу площадь и номер по возрастанию на случай дальнейшей работы
# с этими данными
row_1=np.array(row_1)
row_2=np.array(row_2)
table=np.column_stack((row_1, row_2))
table_data=pd.DataFrame(table)
table_data=np.array(table_data)
b=open('frame5.csv','w')
c=csv.writer(b)
c.writerows(table_data)
b.close()
'''
