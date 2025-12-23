import cv2 
import matplotlib.pyplot as plt 
 
img = cv2.imread(r"C:\Users\suraj\Downloads\dog1.jpg", cv2.IMREAD_GRAYSCALE) 
 
if img is None: 
    raise FileNotFoundError("  Image not found!") 
 
equalized = cv2.equalizeHist(img) 
 
plt.subplot(1,2,1), plt.imshow(img, cmap='gray'), plt.title("Original") 
plt.subplot(1,2,2), plt.imshow(equalized, cmap='gray'), plt.title("Histogram Equalized") 
plt.show()