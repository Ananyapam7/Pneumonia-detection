import os
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('../pneumonia-detection/chest_xray/test/PNEUMONIA/person109_bacteria_522.jpeg')
plt.imshow(img)
plt.show()