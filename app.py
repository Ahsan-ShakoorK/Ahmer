import streamlit as st
import cv2
import numpy as np
import pickle
import os
import csv
import time
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier

# Directory paths
DATA_DIR = 'data'
ATTENDANCE_DIR = 'Attendance'

# Ensure directories exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if not os.path.exists(ATTENDANCE_DIR):
    os.makedirs(ATTENDANCE_DIR)

# Initialize session state variables
if 'face_count' not in st.session_state:
    st.session_state.face_count = 0
if 'capture_key_add' not in st.session_state:
    st.session_state.capture_key_add = 0
if 'capture_key_attendance' not in st.session_state:
    st.session_state.capture_key_attendance = 0

# Function to add faces to the dataset
def add_faces(name):
    faces_data = []
    i = 0

    st.write("Please position your face in front of the camera and press 'Capture' to start.")
    
    while len(faces_data) < 100:
        key = f"capture_add_{st.session_state.capture_key_add}"
        img_file = st.camera_input("Capture", key=key)
        if img_file is not None:
            st.session_state.capture_key_add += 1
            frame = np.array(bytearray(img_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(frame, 1)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = facedetect.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                crop_img = frame[y:y+h, x:x+w, :]
                resized_img = cv2.resize(crop_img, (50, 50))
                if len(faces_data) <= 100 and i % 10 == 0:
                    faces_data.append(resized_img)
                i += 1
                cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)

            st.image(frame, channels="BGR")

    faces_data = np.asarray(faces_data)
    faces_data = faces_data.reshape(100, -1)

    names_path = os.path.join(DATA_DIR, 'names.pkl')
    faces_data_path = os.path.join(DATA_DIR, 'faces_data.pkl')

    if not os.path.exists(names_path):
        names = [name] * 100
        with open(names_path, 'wb') as f:
            pickle.dump(names, f)
    else:
        with open(names_path, 'rb') as f:
            names = pickle.load(f)
        names += [name] * 100
        with open(names_path, 'wb') as f:
            pickle.dump(names, f)

    if not os.path.exists(faces_data_path):
        with open(faces_data_path, 'wb') as f:
            pickle.dump(faces_data, f)
    else:
        with open(faces_data_path, 'rb') as f:
            faces = pickle.load(f)
        faces = np.append(faces, faces_data, axis=0)
        with open(faces_data_path, 'wb') as f:
            pickle.dump(faces, f)

# Function to take attendance
def take_attendance():
    st.write("Please position your face in front of the camera and press 'Capture' to start.")

    key = f"capture_attendance_{st.session_state.capture_key_attendance}"
    img_file = st.camera_input("Capture", key=key)
    if img_file is not None:
        st.session_state.capture_key_attendance += 1
        frame = np.array(bytearray(img_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            st.write("No face detected. Please try again.")
            return

        for (x, y, w, h) in faces:
            crop_img = frame[y:y+h, x:x+w, :]
            resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
            cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
            cv2.putText(frame, "Face Detected", (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

        st.image(frame, channels="BGR")

        names_path = os.path.join(DATA_DIR, 'names.pkl')
        faces_data_path = os.path.join(DATA_DIR, 'faces_data.pkl')

        with open(names_path, 'rb') as w:
            LABELS = pickle.load(w)
        with open(faces_data_path, 'rb') as f:
            FACES = pickle.load(f)

        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(FACES, LABELS)

        output = knn.predict(resized_img)

        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")

        attendance = [str(output[0]), str(timestamp)]

        st.write(f"Attendance taken for {output[0]} at {timestamp}")

        attendance_file = os.path.join(ATTENDANCE_DIR, "attendance.csv")
        exist = os.path.isfile(attendance_file)

        with open(attendance_file, mode='a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not exist:
                writer.writerow(['NAME', 'TIME'])
            writer.writerow(attendance)

st.title("Face Recognition Attendance System")

# Tab for adding faces
st.header("Add Faces")
name = st.text_input("Enter Your Name:")
if st.button("Start Adding Faces"):
    add_faces(name)

# Tab for taking attendance
st.header("Take Attendance")
if st.button("Start Taking Attendance"):
    take_attendance()
