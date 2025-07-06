import cv2               # For image processing
import mediapipe as mp   # For pretrained and honestly really good model of hand tracking
#import torch             # For convolutional neural network
import threading         # For the purposes of creating a 5 second delay and then taking a snap shot while not interrupting video playback
import time              # Needed to measure time
import pandas as pd      # Used for managing the data

#-----------------------------------------Globals-------------------------------------------------

secondsPerSnap = 5
snapready = False

#----------------------------------------Functions------------------------------------------------

def timer(cam):
  global snapready
  startTime = time.time()
  while cam.isOpened():
    t = time.time()
    if t-startTime > secondsPerSnap:
      snapready = True # Set global variable so it can be used in recordData() to trigger an if and grab the data, then set snap back to False
      startTime = t

def signGenerator(): # Since i needed to loop through 0-2 for each sign i fancied making a generator
  while True:
    for i in range(3):
      yield i

def recordData():
  global snapready # For some reason this global doesn't like when it doesnt get global'd like this.

  #The signs to be recorded
  signlist = ["No", "Eagle", "Easily"]
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
          data.append([allLandMarks, signlist[currSign]])
      if snap:
        snap = False
      
      #Add on video message to show what sign to do next.
      cv2.putText(image, f"{signlist[currSign]}", (0,50), font, fontscale, (255,255,255), thickness, cv2.LINE_AA)

      # Flip the image horizontally
      cv2.imshow('MediaPipe Hands', image)
      if cv2.waitKey(5) & 0xFF == 27:
        break
  
  cap.release()
  return data
#-------------------------------------------Main-------------------------------------------------

#Firstly i need to create the neural network and use the GPU if available because why not.
#Start using GPU if available
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
  print("Running on the GPU")
elif device.type == "cpu":
  print("Running on the CPU")
""" # Temp until i setup torch

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
elif training == 1:
  pass # When I get the data recording properly then it's time to setup the torch function. Should be quite standard.

data = pd.DataFrame(data, columns=["landmark Cords", "Sign"])

print(data.head())