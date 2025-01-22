import cv2
from sklearn.pipeline import Pipeline
import joblib
import numpy as np
import pywt

##WAVELET
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

# Load the saved model
model = joblib.load('saved_model.pkl')
# Open the video file
cap = cv2.VideoCapture('video.mp4')

# Check if the video is opened successfully
if not cap.isOpened():
    print("Error opening video file")

# Read and process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_detect = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')
    faces = face_detect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        input = []
        face_roi = gray[y:y+h, x:x+w]
        scaled_raw_img = cv2.resize(frame, (32, 32))
        img_har = w2d(frame, 'db1', 5)
        scaled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scaled_raw_img.reshape(32 * 32 * 3, 1), scaled_img_har.reshape(32 * 32, 1)))
        input.append(combined_img)
        input = np.array(input).reshape(len(input), len(input[0])).astype(float)
        prediction = model.predict(input)
        if prediction == 'chae':  # Adjust this condition based on your model's output
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, 'CHAEWON', (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

        if prediction == 'kazu':  # Adjust this condition based on your model's output
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, 'KAZUHA', (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)


    if ret:
        # Note: You'll need to adjust this line based on how your model is used.

        # Optional: Display the frame with detected faces
        cv2.imshow('Frame', frame)

        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

# When everything done, release the video capture object
cap.release()
# Closes all the frames
cv2.destroyAllWindows()