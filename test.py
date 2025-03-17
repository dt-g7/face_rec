import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt
import os
import shutil
# TESTING
img = cv2.imread('data/testimg/00.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print(gray.shape)
# plt.imshow(gray, cmap='gray')
# plt.show()
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascades/haarcascade_eye.xml')
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
print(faces)
print(faces[0])
print(faces[1])
x, y, w, h = faces[0]

face_img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)





for (x, y, w, h) in faces:
    face_img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = face_img[y:y + h, x:x + w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
# Convert BGR to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_rgb1 = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
# Display image
plt.imshow(img_rgb)
plt.show()
plt.imshow(img_rgb1)
plt.show()