import math
import cv2
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils

def detect_pose(image, pose, display=True):
    output_image = image.copy()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(imageRGB)
    height = image.shape[0]
    width = image.shape[1]
    landmarks = []
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)

        for landmark in results.pose_landmarks.landmark:
            landmarks.append((int(landmark.x * width), int(landmark.y * height), int(landmark.z * width)))

    
    
    return output_image, landmarks

def calculateAngle(landmark1, landmark2, landmark3):
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

    if angle < 0:
        angle += 360

    return angle


def classifyPose(landmarks, output_image, display=False):
    label = 'Unknown pose'
    color = (0, 0, 255)

    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])

    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])

    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])

    # Define angle thresholds for cricket shots
    cover_drive_threshold = 120
    straight_drive_threshold = 90
    defense_threshold = 120

    # Check for each shot
    if left_elbow_angle > cover_drive_threshold and right_elbow_angle > cover_drive_threshold:
        if left_shoulder_angle > 120 :
            if left_knee_angle >=120 and left_knee_angle < 170 :
                label = 'PULL SHOT'
        elif left_elbow_angle >= 120:
            label = 'COVER DRIVE'
    
    
           
    if left_knee_angle <= 90 and right_elbow_angle >=170:
        label = 'SWEEP SHOT'    
        
        
    
    
    if label != 'Unknown pose':
        color = (0, 0, 255)
    cv2.putText(output_image, label, (40, 70), cv2.FONT_HERSHEY_PLAIN,2,color,2)
    
    

    # Additional code for displaying or saving the result
    if display:
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title('output image'); plt.axis('off');

    else:
        return output_image,label
    


def generate_frames():
    video = cv2.VideoCapture(0)
    video.set(3, 1280)
    video.set(4, 960)

    while True:
        ok, frame = video.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape
        # Resize the frame before passing it to detect_pose
        frame_resized = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))
        frame, landmark = detect_pose(frame_resized, pose_video, display=False)

        if landmark:
            # Draw landmarks on the original frame
            frame, _ = classifyPose(landmark, frame, display=False)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    video.release()


# import math
# import cv2
# import numpy as np
# import mediapipe as mp
# from flask import Flask, render_template, Response

# app = Flask(__name__)

# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
# mp_drawing = mp.solutions.drawing_utils

# def detect_pose(image, pose, display=True):
#     output_image = image.copy()
#     imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = pose.process(imageRGB)
#     height = image.shape[0]
#     width = image.shape[1]
#     landmarks = []
#     if results.pose_landmarks:
#         mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
#                                   connections=mp_pose.POSE_CONNECTIONS)

#         for landmark in results.pose_landmarks.landmark:
#             landmarks.append((int(landmark.x * width), int(landmark.y * height), int(landmark.z * width)))

#     return output_image, landmarks

# def calculate_angle(landmark1, landmark2, landmark3):
#     x1, y1, _ = landmark1
#     x2, y2, _ = landmark2
#     x3, y3, _ = landmark3

#     angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

#     if angle < 0:
#         angle += 360

#     return angle

# def classify_pose(landmarks, output_image, display=False):
#     label = 'Unknown pose'
#     color = (0, 0, 255)

#     left_elbow_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
#                                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
#                                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])

#     right_elbow_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
#                                         landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
#                                         landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])

#     left_shoulder_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
#                                           landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
#                                           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

#     right_shoulder_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
#                                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
#                                            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

#     left_knee_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
#                                       landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
#                                       landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

#     right_knee_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
#                                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
#                                        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])

#     # Define angle thresholds for cricket shots
#     cover_drive_threshold = 120
#     straight_drive_threshold = 90
#     defense_threshold = 120

#     # Check for each shot
#     if left_elbow_angle > cover_drive_threshold and right_elbow_angle > cover_drive_threshold:
#         if left_shoulder_angle > 120:
#             if left_knee_angle >= 120 and left_knee_angle < 170:
#                 label = 'PULL SHOT'
#         elif left_elbow_angle >= 120:
#             label = 'COVER DRIVE'

#     if left_knee_angle <= 90 and right_elbow_angle >= 170:
#         label = 'SWEEP SHOT'

#     if label != 'Unknown pose':
#         color = (0, 0, 255)
#     cv2.putText(output_image, label, (40, 70), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

#     # Additional code for displaying or saving the result
#     if display:
#         cv2.imshow('Pose Detection', output_image)
#         cv2.waitKey(1)

#     return output_image, label

# @app.route('/')
# def index():
#     return render_template('index.html')

# def generate_frames():
#     video = cv2.VideoCapture(0)
#     video.set(3, 1280)
#     video.set(4, 960)

#     while True:
#         ok, frame = video.read()
#         if not ok:
#             break

#         frame = cv2.flip(frame, 1)
#         frame_height, frame_width, _ = frame.shape
#         frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))
#         output_frame, landmark = detect_pose(frame, pose, display=False)

#         if landmark:
#             output_frame, _ = classify_pose(landmark, output_frame, display=False)

#         ret, buffer = cv2.imencode('.jpg', output_frame)
#         frame = buffer.tobytes()

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

#     video.release()


