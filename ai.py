import math
import cv2
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(static_image_mode = True , min_detection_confidence = 0.3 , model_complexity = 2)

mp_drawing = mp.solutions.drawing_utils

sample_img = cv2.imread("C:/Users/Sankalp/OneDrive/Desktop/AI/cd1.jpg")
plt.figure(figsize = [10,10])

plt.title("Sample Image");plt.axis('off');plt.imshow(sample_img[:, :, ::-1]);plt.show()

def detctPose(image , pose , display = True):
  output_image = image.copy()
  imageRGB = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
  results = pose.process(imageRGB)
  height = image.shape[0]
  width = image.shape[1]
  landmarks = []
  if results.pose_landmarks:
    mp_drawing.draw_landmarks(image = output_image , landmark_list=results.pose_landmarks , connections = mp_pose.POSE_CONNECTIONS)

    for landmark in results.pose_landmarks.landmark:
      landmarks.append((int(landmark.x * width) , int(landmark.y * height) , (landmark.z * width)))

  if display:
    plt.figure(figsize = [22,22])
    plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off')
    plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off')

    mp_drawing.plot_landmarks(results.pose_world_landmarks,mp_pose.POSE_CONNECTIONS)

  else:
    return output_image,landmarks

image = cv2.imread("C:/Users/Sankalp/OneDrive/Desktop/AI/cd1.jpg")
detctPose(image , pose , display = True)


def calculateAngle(landmark1,landmark2,landmark3):
    x1,y1, _ = landmark1
    x2,y2 ,_ = landmark2
    x3, y3, _ = landmark3
    
    angle = math.degrees(math.atan2(y3-y2,x3-x2)-math.atan2(y1-y2,x1-x2))
    
    if angle <0:
        angle += 360
        
    return angle

angle = calculateAngle((558,326,0),(642,333,0),(718,321,0))
print(f'The calculated angle is {angle}')


def classifyPose(landmarks,output_image,display=False):
  label = 'Unknown pose'
  
  color = (0,0,255)
  
  left_elbow_angle =  calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    
  right_elbow_angle =  calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])
  
  left_shoulder_angle =  calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
  
  right_shoulder_angle =  calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
    
  left_knee_angle =  calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
    
  right_knee_angle =  calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    
  # Define angle thresholds for cricket shots
  cover_drive_threshold = 120
  straight_drive_threshold = 160
  defense_threshold = 90

  # Check for each shot
  if left_elbow_angle > cover_drive_threshold and right_elbow_angle > cover_drive_threshold:
      label = 'Cover Drive'
      color = (0, 255, 0)
  elif left_elbow_angle < straight_drive_threshold and right_elbow_angle < straight_drive_threshold:
      label = 'Straight Drive'
      color = (255, 0, 0)
  elif left_elbow_angle < defense_threshold and right_elbow_angle < defense_threshold:
      label = 'Defense'
      color = (0, 0, 255)

  # Additional code for displaying or saving the result
  if display:
        # Your code for displaying the output image with the label and color
      pass

  return label, color


pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

video = cv2.VideoCapture(0)
video.set(3, 1280)
video.set(4, 960)
cv2.namedWindow('Pose Detection', cv2.WINDOW_NORMAL)


while video.isOpened():
  ok, frame = video.read()
  if not ok:
     continue

  frame = cv2.flip(frame, 1)
  frame_height ,frame_width , _ = frame.shape
  frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))
  frame, landmark = detctPose(frame, pose_video, display=False)

  if landmark:
    frame, _ = classifyPose(landmark, frame, display=False)

  cv2.imshow('Pose Detection', frame)
  k = cv2.waitKey(1) & 0xFF

  if k == 27:
    break

video.release()
cv2.destroyAllWindows()

