import cv2 
import cv2 
import matplotlib.pyplot as plt 
 
img = cv2.imread(r"C:\Users\karti\Downloads\cats.jpg", cv2.IMREAD_GRAYSCALE) 
edges = cv2.Canny(img, 100, 200) 
 
plt.subplot(1,2,1), plt.imshow(img, cmap='gray'), plt.title("Original") 
plt.subplot(1,2,2), plt.imshow(edges, cmap='gray'), plt.title("Edge Detection") 
plt.show()