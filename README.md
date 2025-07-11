# SignLanWriter
An application using CNN's to recognise sign language and show it back to the user on the webcam.

Note, this only works for 1 hand, i have not implemented 2 hands although it would be easily possible with some changes in the training script.

## Installation

After downloading the files, make sure to open a command terminal in that file and run the command ```pip install -r requirements.txt``` This will install all requirements for the script to work.
Ensure you have python 3.8 or later installed, as some imports may not work. There is a warning available in the training script.

## How to

If you want to create your own from scratch it's very simple.

### Step 1

Add the signs you want to record into the "signs.txt" file, simply list them. This file controls what signs are being recognised when recording. And is important for displaying signs when testing.

### Step 2

Run the script ```python "Hand tracking.py"```. This will run the hand tracking script, make sure you have a web cam obviously or this will not work.

When prompted, type 0, this will start your webcam, and when you have your hand on screen you will see it being recognised. 

IMPORTANT NOTE:
This is only designed for 1 hand, 2 hands will result in the attempt being discarded. I could add this in, but I only really wanted to make this as a test for hand recognition.

### Step 3

You will see your words displayed in the corner of the webcam preview. Simply follow the text and perform the sign. When the text changes, the data has been recorded and will wait 5 seconds per snapshot.

### Step 4

After you have collected a sufficient amount of data simply hit esc with the camera window selected and the application will finish. To train, run the script again (Press up on the console for the previous command) and instead of pressing 0, press 1.

This will run the model training and afterwards you will recieve a graph of the training aswell as this, the console will display the training at each epoch.

<img width="636" height="551" alt="image" src="https://github.com/user-attachments/assets/143e32a0-a963-44b1-b33d-7e78a428d1f0" />

Should the result be unsatisfactory. You can play around with the ```lr``` and ```epochs``` until you are satisfied.

### Step 5

Simply run the ```python test.py``` file and your camera will open, when your hand is on display it will attempt to recognise the sign.

And it's just that simple really.
