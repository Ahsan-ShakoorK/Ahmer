import streamlit as st
import cv2
import numpy as np
import pickle
import os
import csv
import time
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier

# Function to add faces to the dataset
def add_faces(name):
    video = cv2.VideoCapture(0)
    facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

    if not video.isOpened():
        st.error("Error: Could not open video source.")
        return

    faces_data = []
    i = 0

    placeholder = st.empty()

    while len(faces_data) < 100:
        ret, frame = video.read()
        if not ret or frame is None:
            st.error("Error: Failed to grab frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            crop_img = frame[y:y+h, x:x+w, :]
            resized_img = cv2.resize(crop_img, (50, 50))
            if len(faces_data) <= 100 and i % 10 == 0:
                faces_data.append(resized_img)
            i += 1
            cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)

        placeholder.image(frame, channels="BGR", use_column_width=True)

    video.release()
    cv2.destroyAllWindows()

    faces_data = np.asarray(faces_data)
    faces_data = faces_data.reshape(100, -1)

    if not os.path.exists('data'):
        os.makedirs('data')

    if 'names.pkl' not in os.listdir('data/'):
        names = [name] * 100
        with open('data/names.pkl', 'wb') as f:
            pickle.dump(names, f)
    else:
        with open('data/names.pkl', 'rb') as f:
            names = pickle.load(f)
        names += [name] * 100
        with open('data/names.pkl', 'wb') as f:
            pickle.dump(names, f)

    if 'faces_data.pkl' not in os.listdir('data/'):
        with open('data/faces_data.pkl', 'wb') as f:
            pickle.dump(faces_data, f)
    else:
        with open('data/faces_data.pkl', 'rb') as f:
            faces = pickle.load(f)
        faces = np.append(faces, faces_data, axis=0)
        with open('data/faces_data.pkl', 'wb') as f:
            pickle.dump(faces, f)

# Function to take attendance
def take_attendance():
    video = cv2.VideoCapture(0)
    facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

    # Load data from pickle files
    with open('data/names.pkl', 'rb') as w:
        LABELS = pickle.load(w)
    with open('data/faces_data.pkl', 'rb') as f:
        FACES = pickle.load(f)

    # Initialize KNN classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(FACES, LABELS)

    # Initialize background image
    imgBackground = cv2.imread("background.png")

    # CSV column names
    COL_NAMES = ['NAME', 'TIME']

    placeholder = st.empty()

    while True:
        ret, frame = video.read()
        if not ret or frame is None:
            st.error("Error: Failed to grab frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            crop_img = frame[y:y+h, x:x+w, :]
            resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
            output = knn.predict(resized_img)

            ts = time.time()
            date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
            timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")

            # Prepare attendance record
            attendance = [str(output[0]), str(timestamp)]

            # Draw rectangles and text on the frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
            cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
            cv2.putText(frame, str(output[0]), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)

        placeholder.image(frame, channels="BGR", use_column_width=True)

        k = cv2.waitKey(1)
        if k == ord('o'):
            time.sleep(5)

            # Check if attendance CSV file exists
            attendance_file = "Attendance/osama.csv"
            exist = os.path.isfile(attendance_file)

            # Write attendance record to CSV
            with open(attendance_file, mode='a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if not exist:
                    writer.writerow(COL_NAMES)
                writer.writerow(attendance)

        if k == ord('q'):
            break

    # Release video capture and close all windows
    video.release()
    cv2.destroyAllWindows()

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
