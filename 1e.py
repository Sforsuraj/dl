import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
 
img = cv2.imread(r"C:\Users\karti\Downloads\cats.jpg", cv2.IMREAD_GRAYSCALE) 
_, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY) 
 
kernel = np.ones((5,5), np.uint8) 
dilation = cv2.dilate(thresh, kernel, iterations=1) 
erosion = cv2.erode(thresh, kernel, iterations=1) 
 
plt.subplot(1,3,1), plt.imshow(thresh, cmap='gray'), plt.title("Thresholded") 
plt.subplot(1,3,2), plt.imshow(dilation, cmap='gray'), plt.title("Dilation") 
plt.subplot(1,3,3), plt.imshow(erosion, cmap='gray'), plt.title("Erosion") 
plt.show()