from flask import Flask, render_template, request, redirect, url_for,Response
from pose_detection import detect_pose
from pose_detection import generate_frames
import os
import cv2
import mediapipe as mp
import numpy as np
import base64
from scipy.spatial import distance
import math

app = Flask(__name__)

UPLOAD_FOLDER = 'static/images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        # Read image directly from the uploaded file
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        frame,landmarks = detect_pose(image, pose, display=False)

        # Create a copy of the original image to draw landmarks and connect them
        output_image = image.copy()

        # Draw landmarks on the image and calculate distances
        # distances = []
        # for i, landmark in enumerate(landmarks):
        #     cv2.circle(output_image, (landmark[0], landmark[1]), 5, (0, 255, 0), -1)
        #     if i < len(landmarks) - 1:
        #         # Calculate Euclidean distance between consecutive landmarks
        #         dist = distance.euclidean(landmarks[i], landmarks[i + 1])
        #         distances.append(dist)
        #         # Display distance on the image
        #         cv2.putText(output_image, f"{dist:.2f}", (landmark[0], landmark[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # # Connect landmarks with lines
        # for i in range(len(landmarks) - 1):
        #     cv2.line(output_image, (landmarks[i][0], landmarks[i][1]), (landmarks[i + 1][0], landmarks[i + 1][1]), (0, 255, 0), 2)
        
    limb_connections = [
        # Legs
        (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
        (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
        (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
        (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
        # Arms
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
        (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
        (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
        # Upper body
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
        (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
        (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
        # Face
        (mp_pose.PoseLandmark.LEFT_EYE, mp_pose.PoseLandmark.LEFT_EAR),
        (mp_pose.PoseLandmark.RIGHT_EYE, mp_pose.PoseLandmark.RIGHT_EAR),
        (mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_EYE),
        (mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.RIGHT_EYE),
        (mp_pose.PoseLandmark.LEFT_EYE, mp_pose.PoseLandmark.RIGHT_EYE),
    ]

    # Create a copy of the original image to draw landmarks and connect specific body parts
    output_image = image.copy()

    # Draw landmarks on the image and calculate distances
    # distances = []
    # for i, landmark in enumerate(landmarks):
    #     cv2.circle(output_image, (landmark[0], landmark[1]), 5, (0, 255, 0), -1)
    #     if i < len(landmarks) - 1:
    #         # Calculate Euclidean distance between consecutive landmarks
    #         dist = distance.euclidean(landmarks[i], landmarks[i + 1])
    #         distances.append(dist)
    #         # Display distance on the image
    #         cv2.putText(output_image, f"{dist:.2f}", (landmark[0], landmark[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    angles = []
    for i in range(len(landmarks) - 2):  # Iterate up to the second-to-last landmark
        point1 = np.array(landmarks[i])
        point2 = np.array(landmarks[i + 1])
        point3 = np.array(landmarks[i + 2])

        # Vectors representing the two sides of the angle
        vector1 = point1 - point2
        vector2 = point3 - point2

        # Calculate the angle in radians
        angle_radians = math.acos(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)))
        cosine_value = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        cosine_value = max(min(cosine_value, 1.0), -1.0)

        angle_radians = math.acos(cosine_value)
        # Convert angle to degrees
        angle_degrees = math.degrees(angle_radians)

        angles.append(angle_degrees)

        # Display angle on the image
        cv2.putText(output_image, f"{angle_degrees:.2f}", (landmarks[i + 1][0], landmarks[i + 1][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        #draw landmark
        cv2.circle(output_image, (landmarks[i + 1][0], landmarks[i + 1][1]), 5, (0, 255, 0), -1)
        
        
    # Connect specific landmarks with lines
    for connection in limb_connections:
        if connection[0].value < len(landmarks) and connection[1].value < len(landmarks):
            point1 = (int(landmarks[connection[0].value][0]), int(landmarks[connection[0].value][1]))
            point2 = (int(landmarks[connection[1].value][0]), int(landmarks[connection[1].value][1]))
            cv2.line(output_image, point1, point2, (0, 255, 0), 2)
    
    # Connect specific landmarks with lines
        # for connection in limb_connections:
        #     # Check if landmarks list has enough elements
        #     if connection[0].value < len(landmarks) and connection[1].value < len(landmarks):
        #         point1 = (int(landmarks[connection[0].value][0]), int(landmarks[connection[0].value][1]))
        #         point2 = (int(landmarks[connection[1].value][0]), int(landmarks[connection[1].value][1]))
        #         cv2.line(output_image, point1, point2, (0, 255, 0), 2)

    # Convert the image to base64 for displaying in HTML
    _, img_encoded = cv2.imencode('.png', output_image)
    img_base64 = 'data:image/png;base64,{}'.format(base64.b64encode(img_encoded).decode())

    return render_template('index.html', image_file=img_base64, distances = angles)


if __name__ == '__main__':
    app.run(debug=True)

