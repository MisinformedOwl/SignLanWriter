import cv2               # For image processing
import mediapipe as mp   # For pretrained and honestly really good model of hand tracking
import torch             # For convolutional neural network
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd      # Used for managing the data
import pickle
import matplotlib.pyplot as plt
from modelnetwork import torchNN

#------------------------Functions-------------------------------

def testing(model: torch.nn) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    cam = cv2.VideoCapture()

    #Setup
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 2
    thickness = 2
    signlist = []
    with open("signs.txt", "r") as file:
        while True:
            word = file.readline()
            if word == "":
                break
            signlist.append(word.rstrip())

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        while cap.isOpened():
            allMarks = []
            success, image = cap.read()
            if not success:
                print("Empty frame")
                continue
            
            sign = ""

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                    if allMarks == 21:
                        raise Exception("TOO MANY HANDS")
                        break
                    for mark in hand_landmarks.landmark:
                        allMarks.append(mark.x)
                        allMarks.append(mark.y)
                        allMarks.append(mark.z)
                if len(allMarks) != 63:
                    print("Too many marks, skipping prediction")
                    continue
                allMarks = torch.tensor(allMarks).to(device)
                allMarks = allMarks.unsqueeze(0)
                predict = model(allMarks)[0]
                predIndex = torch.argmax(predict)
                if predict[predIndex] >= 0.6: # If predicted value has a confidence of 60% or greater
                    sign =  signlist[predIndex-1]
                
            cv2.putText(image, f"{sign}", (0,50), font, fontscale, (255,255,255), thickness, cv2.LINE_AA)
            cv2.imshow("Sign Language Reading", image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    # Largely using the same code as before
    #If i wanted to optimize it i could have this in it's own utility function, but that all seems like hassle for what i'm trying to do here. Not much will change.

def getModel() -> torch.nn:
    with open("model.pk", "rb") as file:
        model = pickle.load(file)
    return model

#---------------------------Main---------------------------------

if __name__ == "__main__":
    model = getModel()
    testing(model)