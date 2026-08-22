# Face Gaze Tracker Demo Setup Instructions

## Contents
* `eye_gaze`: Arduino code to be deployed on Arduino Nano 33 BLE Sense device
* `eye_gaze_listener.py`: Python script to run on terminal that will listen for images and messages sent from Arduino board.

## Prerequisites
The `eye_gaze_listener.py` program requires the following packages

* `pyserial`
* `opencv-python`

It also assumes you have Python 3.13+.

## Instructions

1. Ensure your Arduino board is attached to a TinyML Shield equipped with a OV7675 Camera Module
2. Open the `eye_gaze.ino` file using the Arduino IDE
3. Flash the Arduino board using the sketch code in the `eye_gaze` directory.
4. Take note of the port name of your Arduino board.
  
   <img width="235" height="242" alt="board_port" src="https://github.com/user-attachments/assets/bd33a376-9a1d-4ea8-98ca-5064a5bed4ff" />
5. Close the Arduino IDE.
6. In `eye_gaze_listener.py` double-check that the `PORT` variable is assigned to the correct port name of your Arduino board from step 4.
   
  <img width="834" height="174" alt="code" src="https://github.com/user-attachments/assets/6d99d8dd-55f4-490d-97fd-83ba5461405b" />
  
7. Navigate to the folder containing `eye_gaze_listener.py`
8. Run `python ./eye_gaze_listener.py`
9. Press the button on the TinyML Shield to take a picture.
