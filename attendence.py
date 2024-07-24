# import cv2
# import pickle
# import numpy as np
# import os
# import csv
# import time
# from datetime import datetime
# from win32com.client import Dispatch
# from sklearn.neighbors import KNeighborsClassifier  # Import KNeighborsClassifier

# def take_attendance():
#     speak = Dispatch(("SAPI.SpVoice"))

#     video = cv2.VideoCapture(0)
#     facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

#     # Load data from pickle files
#     with open('data/names.pkl', 'rb') as w:
#         LABELS = pickle.load(w)
#     with open('data/faces_data.pkl', 'rb') as f:
#         FACES = pickle.load(f)

#     # Check the shapes of loaded data
#     print('Shape of Faces matrix --> ', FACES.shape)
#     print('Number of labels --> ', len(LABELS))

#     # Initialize KNN classifier
#     knn = KNeighborsClassifier(n_neighbors=5)
#     knn.fit(FACES, LABELS)

#     # Initialize background image
#     imgBackground = cv2.imread("background.png")

#     # CSV column names
#     COL_NAMES = ['NAME', 'TIME']

#     while True:
#         ret, frame = video.read()
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = facedetect.detectMultiScale(gray, 1.3, 5)
        
#         for (x, y, w, h) in faces:
#             crop_img = frame[y:y+h, x:x+w, :]
#             resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
#             output = knn.predict(resized_img)
            
#             ts = time.time()
#             date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
#             timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
            
#             # Prepare attendance record
#             attendance = [str(output[0]), str(timestamp)]
            
#             # Draw rectangles and text on the frame
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
#             cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
#             cv2.putText(frame, str(output[0]), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
        
#         # Display frame on background image
#         imgBackground[162:162 + 480, 55:55 + 640] = frame
#         cv2.imshow("Frame", imgBackground)
        
#         k = cv2.waitKey(1)
#         if k == ord('o'):
#             speak.Speak("Attendance Taken..")
#             time.sleep(5)
            
#             # Check if attendance CSV file exists
#             attendance_file = "Attendance\\osama.csv"
#             exist = os.path.isfile(attendance_file)
            
#             # Write attendance record to CSV
#             with open(attendance_file, mode='a', newline='') as csvfile:
#                 writer = csv.writer(csvfile)
#                 if not exist:
#                     writer.writerow(COL_NAMES)
#                 writer.writerow(attendance)
        
#         if k == ord('q'):
#             break

#     # Release video capture and close all windows
#     video.release()
#     cv2.destroyAllWindows()
