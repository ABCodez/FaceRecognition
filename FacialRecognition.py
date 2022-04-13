import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Get path to Images folder and load images into a list
path = 'Images'
images = []
names = []
imgList = os.listdir(path)
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


# Security footage log
def SecureLog(name):
    with open('SecurityLog.csv', 'r+') as f:  # open log and write to it
        logList = f.readlines()
        idList = []
        for eachLine in logList:
            entry = eachLine.split(',')  # format entry
            idList.append(entry[0])
        if name not in idList:  # add name and time only ONCE (can be changed to have re-occurring name+time)
            current = datetime.now()
            date = current.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{date}')



# output if faces are recognized
recognizedFaces = encode(images)
print('Faces Recognized!')

# initialize video capture
vc = cv2.VideoCapture(1)

while True:
    success, img = vc.read()
    imgSize = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # resize video by factor of 4 to speed up process
    imgSize = cv2.cvtColor(imgSize, cv2.COLOR_BGR2RGB)  # Convert image to rgb

    initFrame = face_recognition.face_locations(imgSize)  # find all faces that may be present within frame
    encodeFrame = face_recognition.face_encodings(imgSize, initFrame)  # find encodings within video capture

    # loop through all faces found in frame (grab a face from frame as well as its encoding)
    for encodeFace, locateFace in zip(encodeFrame, initFrame):
        parallel = face_recognition.compare_faces(recognizedFaces, encodeFace)  # compare recognized faces w/ encodeFace
        distance = face_recognition.face_distance(recognizedFaces, encodeFace)  # find face distance attributes
        print(distance)
        parallelIndex = np.argmin(distance)  # find the lowest distance as it will be our best match to the current face

        # Label recognized and unknown faces (if the distance is greater than 2.0 == person is unknown (alter #))
        if parallel[parallelIndex] > 0.8:
            faceName = names[parallelIndex].upper()
            print(faceName)
        else:
            faceName = "Unknown"
            print(faceName)

        # bounding box for image
        color = (0, 255, 0)
        y1, x2, y2, x1 = locateFace
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4  # scale video back to original size
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)  # draw rectangle
        cv2.rectangle(img, (x1, y2 - 30), (x2, y2), color, cv2.FILLED)
        cv2.putText(img, faceName, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        SecureLog(faceName)

    # display final result
    cv2.imshow("Facial Recognition - ABCodez", img)
    cv2.waitKey(1)
