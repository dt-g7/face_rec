import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt
import os
import shutil
import pywt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
import json

# TESTING
# img = cv2.imread('./testimg/0.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # print(gray.shape)
# # plt.imshow(gray, cmap='gray')
# # plt.show()
# face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascades/haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('./haarcascades/haarcascades/haarcascade_eye.xml')
# faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
# x, y, w, h = faces[0]
#
# face_img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
#

# plt.imshow(face_img)
# plt.show()


####################### BOX AROUND EYES AND FACE
# for (x, y, w, h) in faces:
#     face_img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
#     roi_gray = gray[y:y + h, x:x + w]
#     roi_color = face_img[y:y + h, x:x + w]
#     eyes = eye_cascade.detectMultiScale(roi_gray)
#     for (ex, ey, ew, eh) in eyes:
#         cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
# plt.figure()
# plt.imshow(face_img, cmap='gray')
# plt.show()
# plt.imshow(roi_color, cmap='gray')
# plt.show()

face_cascade = cv2.CascadeClassifier('../haarcascades/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('../haarcascades/haarcascades/haarcascade_eye.xml')


def get_cropped_image_if_2_eyes(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error reading image from path: {image_path}")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            return roi_color


# c = get_cropped_image_if_2_eyes('./testimg/0.jpg')
# plt.imshow(c)
# plt.show()
path_to_data = '../data/testimg'
path_to_cr_data = '../data/testimg/cropped'
img_dirs = []

# create crop path if needed and img dirs
for entry in os.scandir(path_to_data):
    if entry.is_dir():
        img_dirs.append(entry.path)
if os.path.exists(path_to_cr_data):
    shutil.rmtree(path_to_cr_data)
os.makedirs(path_to_cr_data)

# two vars below are useful later on
cropped_img_dirs = []
file_names_dict = {}
for img_dir in img_dirs:  # for each img directory we are gunna go through and create paths & add paths to dict
    count = 1
    name = os.path.basename(img_dir)
    print(name)

    file_names_dict[name] = []

    for entry in os.scandir(img_dir):  # adding cropped images to cropped dir
        roi_color = get_cropped_image_if_2_eyes(entry.path)
        if roi_color is not None:
            cropped_folder = os.path.join(path_to_cr_data, name)
            if not os.path.exists(cropped_folder):
                os.makedirs(cropped_folder)
                cropped_img_dirs.append(cropped_folder)
                print('Generating cropped images in folder: ', cropped_folder)
            cropped_file_name = name + str(count) + ".png"
            cropped_file_path = os.path.join(cropped_folder, cropped_file_name)
            cv2.imwrite(cropped_file_path, roi_color)
            file_names_dict[name].append(cropped_file_path)
            count += 1


## WAVELET

def w2d(img, mode='haar', level=1):
    imArray = img
    # Datatype conversions
    # convert to grayscale
    imArray = cv2.cvtColor(imArray, cv2.COLOR_RGB2GRAY)
    # convert to float
    imArray = np.float32(imArray)
    imArray /= 255;
    # compute coefficients
    coeffs = pywt.wavedec2(imArray, mode, level=level)

    # Process Coefficients
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0;

    # reconstruction
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)

    return imArray_H


class_dict = {'chae': 0,
              'kazu': 1}
x = []
y = []
for name, training_files in file_names_dict.items():
    for training_img in training_files:
        img = cv2.imread(training_img)
        if img is None:
            continue
        scaled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scaled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scaled_raw_img.reshape(32 * 32 * 3, 1), scaled_img_har.reshape(32 * 32, 1)))
        x.append(combined_img)
        y.append(name)

x = np.array(x).reshape(len(x), len(x[0])).astype(float)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel='rbf', C=10))])
p = pipe.fit(x_train, y_train)
pipe.score(x_test, y_test)
print(classification_report(y_test, pipe.predict(x_test)))

joblib.dump(p, '../models/saved_model.pkl')

with open("../config/class_dictionary.json", "w") as f:
    f.write(json.dumps(class_dict))
