import os
import cv2
import numpy as np
from img_aug import Img_aug

#현재 경로를 갖고와서 디렉토리 변경
currentPath = os.getcwd()
print(currentPath)
os.chdir(currentPath)

aug = Img_aug() #데이터 증강 class 선언 
augment_num = 40 #증강결과로 출력되는 이미지의 갯수 선언 
name = 'barbara'
save_path = 'output/' + name 

#불투명이미지로 교체
img = cv2.imread('bser/Barbara_Mini.jpg') 

images_aug = aug.seq.augment_images([img for i in range(augment_num)]) 

for num,aug_img in enumerate(images_aug) : 
    cv2.imwrite(save_path + '/' + name +'_{}.jpg'.format(num),aug_img) 
    
print('Complete augmenting images')