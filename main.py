import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
import mediapipe as mp
from sklearn.tree import DecisionTreeClassifier
from utils import calculate_angle

countLeft = 0
countRight = 0
prevActionLeft = False
prevActionRight = False

# Download Dataset from: kagglehub.dataset_download("trainingdatapro/pose-estimation")
# Path ke dataset
dataset_path = "body-plot"
xml_path = "dannotations.xml"
image_dir = os.path.join(dataset_path, "PoseEstimation", "PE")

# Persiapan data training
X_train = []
y_train = []

# Parsing file XML
tree = ET.parse(xml_path)
root = tree.getroot()

# Loop melalui setiap gambar dalam file XML
for image_element in root.findall("image"):
    image_name = image_element.get("name")
    image_path = os.path.join(image_dir, image_name)
    keypoints = []

    # Ekstrak elemen <points>
    for points_element in image_element.findall("points"):
        points_str = points_element.get("points")
        label = points_element.get("label")
        if points_str is not None and label is not None:
            points = [float(coord) for coord in points_str.split(",")]
            keypoints.extend(points)

    # Validasi keypoints dan hitung sudut
    try:
        if len(keypoints) >= 12:
            right_angle = calculate_angle(keypoints[0:2], keypoints[2:4], keypoints[4:6])
            left_angle = calculate_angle(keypoints[6:8], keypoints[8:10], keypoints[10:12])
            X_train.append([right_angle, left_angle])
            y_train.append(int(label))
    except IndexError as e:
        print(f"Error processing keypoints: {e}")

# Validasi data training
if len(X_train) > 0 and len(y_train) > 0:
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Latih model Decision Tree
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
else:
    print("Error: Training data is empty.")
    exit()
# Latih model Decision Tree
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Inisialisasi MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Mulai capture webcam
cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # BGR ke RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark
            # Ambil titik COCO: shoulder, elbow, wrist (kanan & kiri)
            r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * frame.shape[1],
                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame.shape[0]]
            r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x * frame.shape[1],
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y * frame.shape[0]]
            r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x * frame.shape[1],
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * frame.shape[0]]
            l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * frame.shape[1],
                          landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame.shape[0]]
            l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x * frame.shape[1],
                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y * frame.shape[0]]
            l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * frame.shape[1],
                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * frame.shape[0]]

            # Hitung sudut siku kanan & kiri
            right_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
            left_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)

            # Prediksi aksi menggunakan Decision Tree
            action = clf.predict([[right_angle, left_angle]])[0]
            actionRight = (right_angle < 55)
            actionLeft = (left_angle < 55)

            # Perhitungan untuk sisi kanan
            if actionRight == 1 and not prevActionRight:
                countRight += 1
            prevActionRight = (actionRight == 1)

            # Perhitungan untuk sisi kiri
            if actionLeft == 1 and not prevActionLeft:
                countLeft += 1
            prevActionLeft = (actionLeft == 1)

            # Total count
            totalCount = countRight + countLeft

            # Label untuk ditampilkan
            label = f"Right: {countRight}, Left: {countLeft}, Total: {totalCount}"

            # Tampilkan hasil
            cv2.putText(image, f"Right Elbow: {int(right_angle)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(image, f"Left Elbow: {int(left_angle)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(image, f"Aksi: {label}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        except Exception as e:
            pass

        # Gambar pose
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Barbell Detection', image)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            countRight = 0
            countLeft = 0
            totalCount = 0
            prevActionRight = False
            prevActionLeft = False

cap.release()
cv2.destroyAllWindows()
