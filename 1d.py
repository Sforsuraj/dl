import cv2 
import matplotlib.pyplot as plt 
import numpy as np 
img = cv2.imread(r"C:\Users\karti\Downloads\cats.jpg", cv2.IMREAD_GRAYSCALE) 
flip = cv2.flip(img, 1) 
rows, cols = img.shape 
M = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1) 
rotated = cv2.warpAffine(img, M, (cols, rows)) 
 
plt.subplot(1,3,1), plt.imshow(img, cmap='gray'), plt.title("Original") 
plt.subplot(1,3,2), plt.imshow(flip, cmap='gray'), plt.title("Flipped") 
plt.subplot(1,3,3), plt.imshow(rotated, cmap='gray'), plt.title("Rotated") 
plt.show()