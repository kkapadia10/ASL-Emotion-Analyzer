import cv2
import mediapipe as mp
import numpy as np
import torch
from transformers import pipeline
import torch.nn as nn

# Define the neural network architecture
class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Load the hand sign recognition model
model_path = './model.pth'

state_dict = torch.load(model_path)
input_size = state_dict['fc1.weight'].shape[1]
model = SimpleNN(input_size=input_size, num_classes=20)

model.load_state_dict(torch.load(model_path))
model.eval()  # Set the model to evaluation mode

classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

cap = cv2.VideoCapture(1)  # Adjust camera index if necessary

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, max_num_hands=1)

labels_dict = {
    0: 'SPACE',
    1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I',
    10: 'J', 11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'O', 16: 'P', 17: 'Q', 18: 'R', 19: 'S',
}

print("labels_dict used for prediction:", labels_dict)

# Initialize variables to accumulate signed words
signed_words = []
word_count = 0

# Initialize sentence
sentence = ""
sentiment = ""

def perform_sentiment_analysis(sentences):
    model_outputs = classifier(sentences)
    first_emotion_label = model_outputs[0][0]['label']
    return first_emotion_label

# Initialize ready_to_sign variable
ready_to_sign = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    instructions = "Get your hand in frame! Press 'S' to capture letter once you are in position, 'Q' to quit, 'C' to clear."
    cv2.putText(frame, instructions, (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "Sentence: " + sentence, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "Sentiment: " + sentiment, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

    if key == ord('s'):
        ready_to_sign = True

    if key == ord('c'):
        sentence = ""


    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS)

    if results.multi_hand_landmarks:
        data_aux = []
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                data_aux.extend([lm.x, lm.y])

        data_aux = np.array(data_aux).reshape(1, -1)
        if data_aux.shape[1] < 42:
            data_aux = np.pad(data_aux, ((0, 0), (0, 42 - data_aux.shape[1])), 'constant')

        input_tensor = torch.tensor(data_aux, dtype=torch.float32)
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)

            if (12 <= predicted.item() <= 19):
                predicted_character = labels_dict.get(predicted.item() - 10, "Unknown")
            elif(2 <= predicted.item() <= 11):
                predicted_character = labels_dict.get(predicted.item() + 8, "Unknown")
            else:
                predicted_character = labels_dict.get(predicted.item(), "Unknown")

            ready_to_add_space = False  # Initialize this flag outside the loop

            # Inside your loop where you process the predicted character
            if ready_to_sign:
                if predicted_character == "SPACE":
                    if not ready_to_add_space:  # Only add space if flag is False
                        sentence += " "
                        ready_to_add_space = True  # Prevent further spaces until reset
                else:
                    signed_words.append(predicted_character)
                    word_count += 1
                    sentence += predicted_character
                    ready_to_add_space = False  # Allow adding space on next recognized space gesture
                ready_to_sign = False

            sentiment = perform_sentiment_analysis(sentence)
            # print(f"Sentiment: {sentiment}") # Used for debugging

        cv2.putText(frame, "Predicted Character: " + predicted_character, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)

cap.release()
cv2.destroyAllWindows()
