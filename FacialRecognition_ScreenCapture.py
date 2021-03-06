import cv2
import numpy as np
import face_recognition
import os
from PIL import ImageGrab

# Get path to Images folder and load images into a list
path = 'Images'
images = []
names = []
imgList = os.listdir(path)
# print(imgList)
# append images into the list
for faces in imgList:
    initImg = cv2.imread(f'{path}/{faces}')
    images.append(initImg)
    names.append(os.path.splitext(faces)[0])  # Output name of image without the file type ('jake.jpg' --> 'jake')


# Encode Images
def encode(images):
    encodedList = []  # final encoded images to be stored in this list
    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert image to rgb
        findEncode = face_recognition.face_encodings(image)[0]  # find encodings within image
        encodedList.append(findEncode)  # append encoded image to our list
    return encodedList


# output if faces are recognized
recognizedFaces = encode(images)
print('Faces Recognized!')


# initialize screen capture
def captureScreen(bbox=(300, 300, 700 + 300, 550 + 300)):
    capScr = np.array(ImageGrab.grab(bbox))
    capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
    return capScr


vc = cv2.VideoCapture(1)

while True:
    success, img = vc.read()
    img = captureScreen()
    imgSize = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # resize video by factor of 4 to speed up process
    imgSize = cv2.cvtColor(imgSize, cv2.COLOR_BGR2RGB)  # Convert image to rgb

    initFrame = face_recognition.face_locations(imgSize)  # find all faces that may be present within frame
    encodeFrame = face_recognition.face_encodings(imgSize, initFrame)  # find encodings within video capture

    # loop through all faces found in frame (grab a face from frame as well as its encoding)
    for encodeFace, locateFace in zip(encodeFrame, initFrame):
        parallel = face_recognition.compare_faces(recognizedFaces, encodeFace)  # compare recognized faces w/ encodeFace
        distance = face_recognition.face_distance(recognizedFaces, encodeFace)  # find face distance attributes
        # print(distance)
        parallelIndex = np.argmin(distance)  # find the lowest distance as it will be our best match to the current face

        # Label recognized and unknown faces (if the distance is greater than 2.0 == person is unknown (alter #))
        if parallel[parallelIndex] < 1.5:
            faceName = names[parallelIndex].upper()
            # print(faceName)
        else:
            faceName = "Unknown"

        # bounding box for image
        color = (0, 255, 0)
        y1, x2, y2, x1 = locateFace
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4  # scale video back to original size
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)  # draw rectangle
        cv2.rectangle(img, (x1, y2 - 30), (x2, y2), color, cv2.FILLED)
        cv2.putText(img, faceName, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

    # display final result
    cv2.namedWindow("Facial Recognition - ABCodez", cv2.WINDOW_NORMAL)
    cv2.imshow("Facial Recognition - ABCodez", img)
    cv2.waitKey(1)
