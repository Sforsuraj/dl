import cv2 
import matplotlib.pyplot as plt 
 
img = cv2.imread(r"C:\Users\karti\Downloads\cats.jpg", cv2.IMREAD_GRAYSCALE) 
_, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY) 
 
plt.subplot(1,2,1), plt.imshow(img, cmap='gray'), plt.title("Original") 
plt.subplot(1,2,2), plt.imshow(thresh, cmap='gray'), plt.title("Thresholded") 
plt.show()