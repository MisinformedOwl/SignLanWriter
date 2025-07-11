import cv2               # For image processing
import mediapipe as mp   # For pretrained and honestly really good model of hand tracking
import torch             # For convolutional neural network
from torch.utils.data import DataLoader, TensorDataset
import threading         # For the purposes of creating a 5 second delay and then taking a snap shot while not interrupting video playback
import time              # Needed to measure time
import pandas as pd      # Used for managing the data
import sys               # For use in detecting the python version, no idea what version i'd need. But it's better they run the same or a later version assuming nothing gets depricated thats important
import pickle
import matplotlib.pyplot as plt
from modelnetwork import torchNN

#Warn user theit version of python may not work with this script.
if sys.version_info < (3,8,0):
  print(f"WARNING: some packages may require python version 3.8 or higher. procede at your own bugs.")

#-----------------------------------------Globals-------------------------------------------------

secondsPerSnap = 5
snapready = False

#-----------------------------------------Training------------------------------------------------

def train():
  #Setting up training data to be a torch tensor and under the device
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  if device.type == "cuda":
    print("Running on the GPU")
  elif device.type == "cpu":
    print("Running on the CPU")
  
  dataloader = grabData(device)

  #Create the model
  model = torchNN().to(device)
  print(model)

  lr = 0.005
  epochs = 200
  loss_crit = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  history = []

  model.train()

  print("Training".center(50, "-"))
  print("")
  for epoch in range(epochs):
    print("\r", end = "")
    print(f"Epochs: {epoch+1}/{epochs}".center(50, "-"), end="")
    optimizer.zero_grad()
    loss = 0
    total = 0
    correct = 0
    for marks, label in dataloader:
      result = model(marks)
      loss+= loss_crit(result,label)
      for r, l in zip(result, label):
        r = torch.argmax(r)
        total+=1
        if r == l:
          correct+=1
    
    loss.backward()
    optimizer.step()
    history.append(correct/total)
  
  with open("model.pk", "wb") as file:
    pickle.dump(model, file)
  
  for h in range(len(history)):
    print(f"{h+1}: {history[h]}")
  
  plt.plot(history)
  plt.show()

#----------------------------------------Functions------------------------------------------------


def grabData(device):
  """
  This function very simply grabs the data that has been recorded if it is stored. It will also 
  undergo preprocessing before being sent back to the training function
  """
  with open("data.pk", "rb") as file:
    data = pickle.load(file)
  
  #Firstly, seperate the hand positions from the labels
  labels = []
  landmarks = []
  for d in data:
    if len(d[0]) > 21:
      continue
    labels.append(d[1])
    landmarks.append(d[0])
  #clear up data
  del data

  #Assign data into tensors and set the device to the device being used.
  dataLoader = DataLoader(TensorDataset(torch.tensor(landmarks).to(device),torch.tensor(labels).to(device)), batch_size=16, shuffle=True)

  return dataLoader


def timer(cam):
  """
  Simple timer function that runs on a thread and ends when the camera closes.
  Responcible for telling the recordData function when it can get the data by triggering a flag.
  """
  global snapready
  startTime = time.time()
  while cam.isOpened():
    t = time.time()
    if t-startTime > secondsPerSnap:
      snapready = True # Set global variable so it can be used in recordData() to trigger an if and grab the data, then set snap back to False
      startTime = t


def signGenerator(): # Since i needed to loop through 0-2 for each sign i fancied making a generator
  """
  Fancied making a generator for my signlanguage selector, instead of making a while loop where i use 
  modulation on a value to keep it between 0-2
  """
  while True:
    for i in range(3):
      yield i

#------------------------------------Data Collection---------------------------------------------


def recordData():
  """
This method manages opening the camera and grabbing the data.
It does this by displaying a word on the camera screen which tells the user what sign to perform.
Then, after 5 seconds, which is ample time to position around the room to gather a variety of 
data it will take the cordinates of all landmarks of the hand which will be used as the training data.
"""
  global snapready # For some reason this global doesn't like when it doesnt get global'd like this.

  #The signs to be recorded
  signlist = []
  with open("signs.txt", "r") as file:
      while True:
          word = file.readline()
          if word == "":
              break
          signlist.append(word.rstrip())
  currSign = 0
  snap = False

  font = cv2.FONT_HERSHEY_SIMPLEX
  fontscale = 2
  thickness = 2

  mp_drawing = mp.solutions.drawing_utils
  mp_drawing_styles = mp.solutions.drawing_styles
  mp_hands = mp.solutions.hands
  cap = cv2.VideoCapture(0)

  #Threading for timer
  thread = threading.Thread(target = timer, args=(cap,), daemon=True)
  thread.start()

  data = [] # [xcord,ycord,zcord] of hand landmarks
  with mp_hands.Hands(
      model_complexity=0,
      min_detection_confidence=0.5, #The minimum valueto be tracked
      min_tracking_confidence=0.5) as hands:
    
    #Initialise generator
    signGen = signGenerator()
    
    while cap.isOpened(): # While webcam is on
      success, image = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        continue
      
      #This fucntion is used to get around the potential for snap becoming true during runtime of the hand, meaning i may end up losing some fingers. Which is not ideal.
      #Now it will see if it's been 5 seconds, and then internally set snap so it gets the whole data.
      if snapready:
        snap = True
        snapready = False
        currSign = next(signGen)

      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Make the image colour RGB for the purposes of hand recognition
      results = hands.process(image) # This finds the hands

      # Draw the hand annotations on the image
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

      if results.multi_hand_landmarks:
        allLandMarks = [] # Used to group all handmarks before adding them to the data list
        for hand_landmarks in results.multi_hand_landmarks:
          mp_drawing.draw_landmarks( # This tool draws the points
              image,                 # Using the image to draw on
              hand_landmarks,        # The landmarks recorded in results
              mp_hands.HAND_CONNECTIONS,
              mp_drawing_styles.get_default_hand_landmarks_style(),
              mp_drawing_styles.get_default_hand_connections_style())
          if snap:
            for id, mark in enumerate(hand_landmarks.landmark):
              allLandMarks.append([mark.x, mark.y, mark.z])
        if snap:
          data.append([allLandMarks, currSign])
      if snap:
        snap = False
      
      #Add on video message to show what sign to do next.
      cv2.putText(image, f"{signlist[currSign]}", (0,50), font, fontscale, (255,255,255), thickness, cv2.LINE_AA)

      cv2.imshow('MediaPipe Hands', image)
      if cv2.waitKey(5) & 0xFF == 27: # esc pressed
        break
  
  cap.release()
  return data
#-------------------------------------------Main-------------------------------------------------

#Check if user wants to record images or train the data.
training = 0
while True:
  user = input("Do you want to record images (Only images with hands recognised will be recorded)\nOr train the data?\n0 - 1: ")
  if user == "1":
    training = 1
    break
  elif user == "0":
    training = 0
    break
  else:
    print("Incorrect input, try again.\n")

if training == 0:
  data = recordData()
  with open("data.pk", "wb") as file:
    pickle.dump(data, file)
elif training == 1:
  train()

