import os
import matplotlib.pyplot as plt
import cv2

# img = plt.imread('../chest_xray/test/PNEUMONIA/person109_bacteria_527.jpeg')
# plt.imshow(img)
# plt.show()

fig = plt.figure()

img1 = cv2.imread('../chest_xray/test/PNEUMONIA/person109_bacteria_527.jpeg',0)
img2 = cv2.imread('../chest_xray/test/NORMAL/IM-0001-0001.jpeg',0)
img3 = cv2.imread('../chest_xray/test/NORMAL/IM-0003-0001.jpeg',0)
img4 = cv2.imread('../chest_xray/test/PNEUMONIA/person3_virus_17.jpeg',0)


plt.subplot(2, 2, 1)
plt.hist(img1.ravel(),256,[0,256])
plt.subplot(2, 2, 2)
plt.hist(img2.ravel(),256,[0,256])
plt.subplot(2, 2, 3)
plt.hist(img3.ravel(),256,[0,256])
plt.subplot(2, 2, 4)
plt.hist(img4.ravel(),256,[0,256])

#hist = cv2.calcHist([img],[0],None,[256],[0,256])
plt.show() 