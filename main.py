import cv2
import mediapipe as mp
import numpy as np
from utils import calculate_angle
from sklearn.tree import DecisionTreeClassifier

# Inisialisasi MediaPipe Pose
drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Decision Tree sederhana (dummy, perlu dilatih dengan data nyata)
# Misal threshold sudut siku < 90 derajat = angkat beban
clf = DecisionTreeClassifier()
# Data dummy: [sudut_siku_kanan, sudut_siku_kiri], label: 0 (tidak angkat), 1 (angkat)
X_train = [[160, 160], [80, 160], [160, 80], [75, 75]]
y_train = [0, 1, 1, 1]
clf.fit(X_train, y_train)

countLeft = 0
countRight = 0
prevActionLeft = False
prevActionRight = False

def detect_barbell_action(angles):
    # angles: [right_elbow_angle, left_elbow_angle]
    return clf.predict([angles])[0]

# Mulai capture webcam
cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.6) as pose:
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
            r_shoulder = [landmarks[12].x * frame.shape[1], landmarks[12].y * frame.shape[0]]
            r_elbow = [landmarks[14].x * frame.shape[1], landmarks[14].y * frame.shape[0]]
            r_wrist = [landmarks[16].x * frame.shape[1], landmarks[16].y * frame.shape[0]]
            l_shoulder = [landmarks[11].x * frame.shape[1], landmarks[11].y * frame.shape[0]]
            l_elbow = [landmarks[13].x * frame.shape[1], landmarks[13].y * frame.shape[0]]
            l_wrist = [landmarks[15].x * frame.shape[1], landmarks[15].y * frame.shape[0]]

            # Hitung sudut siku kanan & kiri
            right_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
            left_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)

            # Decision tree: detect barbell lifting
            action = detect_barbell_action([right_angle, left_angle])
            actionRight = (right_angle < 90)
            actionLeft = (left_angle < 90)

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
            drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Barbell Detection', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(10) & 0xFF == ord('r'):
            countRight = 0
            countLeft = 0
            totalCount = 0

cap.release()
cv2.destroyAllWindows()