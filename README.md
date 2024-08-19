# Travel Detection

## Demo Video
[Link](https://drive.google.com/file/d/11RtiZ6z89I6QOKA9DQJCgK8H-CPVFWta/view)


## Purpose
Basketball has various rules that must be followed, one of which is the traveling violation, where a player takes more than two steps without dribbling the ball. Detecting this violation is typically done manually by referees, which is prone to human error and limited perspective. This study aimed to develop a real-time system for detecting traveling violations in basketball games using a deep learning method based on the YOLOv8 algorithm. The system was designed to identify the ball and player poses, then count the player's steps to detect violations. The process was carried out in real-time using a smartphone camera and adequate hardware. The results of the study indicate that the system successfully optimized processing speed and improved detection accuracy, effectively assisting referees in making more accurate decisions and reducing errors that could disadvantage teams. The system demonstrated reliable performance under various environmental conditions, real-time detection capabilities, and adaptability to different perspectives


## Setup
1. Clone project
2. Open project in VSCode
3. Create a virtual environment: `python -m venv myvenv`
4. Activate virtual environment: `\myvenv\Scripts\activate`
5. Install ultralytics package: `pip install ultralytics`
6. Run Python scripts `travel_detectv2.py` 
7. Change the input of the video to either your webcam (`cv2.VideoCapture(0)`)
