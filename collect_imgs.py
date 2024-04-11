"""
This class is responsible for capturing ASL signs that are used to train
a machine learning model. It captures a set amount of camera frames for each
specified category (word/letter) using the openCV library.
"""

import os
import cv2

# ensures data directory exists, or creates one if it does not exist.
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# defines number of pictures (unique datapoints) to take from data for each class.
number_of_classes = 27
dataset_size = 300

# openCV function to capture camera footage
# You might need to play around with this number depending on your system
# Windows typically 0, Mac typically 1.
cap = cv2.VideoCapture(1)


for j in range(number_of_classes):
    # checks if current word/letter exists in data folder. creates directory if not present
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    done = False
    # captures footage from camera in continous loop (allows for recording and data inputting)
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    # captures each frame into the directory as a datapoint for the model to train from
    # gets dataset_size number of frames
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

        counter += 1

# closes openCV windows (camera in this case)
cap.release()
cv2.destroyAllWindows()
