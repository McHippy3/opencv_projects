import cv2
import time
import mediapipe as mp
import numpy as np
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_FINGER_MCP = 5
INDEX_FINGER_PIP = 6
INDEX_FINGER_DIP = 7
INDEX_FINGER_TIP = 8
MIDDLE_FINGER_MCP = 9
MIDDLE_FINGER_PIP = 10
MIDDLE_FINGER_DIP = 11
MIDDLE_FINGER_TIP = 12
RING_FINGER_MCP = 13
RING_FINGER_PIP = 14
RING_FINGER_DIP = 15
RING_FINGER_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)

# Set width and height of cam
HEIGHT = 480
WIDTH = 640
cap.set(3, HEIGHT)
cap.set(4, WIDTH)
last_time = 0

thumb_last_x = None
thumb_last_y = None
index_last_x = None
index_last_y = None
last_dist = None

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        # Read frame
        success, frame = cap.read()

        # Process frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            results = holistic.process(image)
        except Exception as e:
            holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Framerate
        current_time = time.time()
        fps = 1/(current_time-last_time)
        last_time = current_time
        cv2.putText(image, f'FPS: {int(fps)}', (10,30), cv2.FONT_HERSHEY_COMPLEX, 
                    1, (255, 0, 0), 3)

        # Track left hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=2),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=1)
                                 )
        lm = results.left_hand_landmarks
        if lm:
            lm = lm.landmark
            thumb_x, thumb_y = lm[THUMB_TIP].x, lm[THUMB_TIP].y
            index_x, index_y = lm[INDEX_FINGER_TIP].x, lm[INDEX_FINGER_TIP].y

            # Draw line from index to thumb
            if thumb_x and index_x:
                thumb_x, thumb_y = int(thumb_x * WIDTH), int(thumb_y * HEIGHT)
                index_x, index_y = int(index_x * WIDTH), int(index_y * HEIGHT)
                cv2.line(image, (thumb_x, thumb_y),
                                (index_x, index_y), 
                                (0, 255, 0), 5)
                dist = math.sqrt((thumb_x - index_x)**2 + (thumb_y - index_y)**2)
                # Set volume based on ratio or current distance to last distance
                if last_dist:
                    cur_vol = int(round(volume.GetMasterVolumeLevelScalar() * 100))
                    if dist > last_dist and cur_vol == 0:
                        cur_vol = 1
                    new_vol = min(1, (cur_vol * dist/last_dist)/100.0) if last_dist != 0 else 0.0
                    volume.SetMasterVolumeLevelScalar(new_vol, None)
                last_dist = dist
            
            thumb_last_x, thumb_last_y = thumb_x, thumb_y
            index_last_x, index_last_y = index_x, index_y

        cv2.imshow('Webcam', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()