"""
This file goes through each word/letter that the model will train on
and captures the hand coordinates of each valid image.
This data is stored in a pickle to be later used for training.
"""

import os
import pickle
import mediapipe as mp
import cv2

# imports modules from mediapipe that are related to hand tracking
# in particular, maps 21 points to each hand.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# this is the declaration of the hand object used for the actual detection
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

# traverses through each category in the data directory (letters/words to be assigned)
for dir_ in os.listdir(DATA_DIR):
    if dir_ != ".DS_Store":

        # within each word to be mapped, this traverses through all the associated images
        for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
            # stores landmark data for each image
            data_aux = []

            # converts image into rgb format as mediapipe requires rgb format for processing
            # then stores handtracking data in results
            img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            # check to ensure hand landmarks are present in current image
            if results.multi_hand_landmarks:
                # extracts x,y coords of each hand landmark detected
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x)
                        data_aux.append(y)
                # landmarks are stored into data list
                data.append(data_aux)
                labels.append(dir_)

# serializes data landmarks collected into pickle object (bytestream)
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
