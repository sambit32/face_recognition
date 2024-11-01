import cv2
import numpy as np
import os
import pickle

# Ensure the data directory exists
data_dir = './data/'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)


# Define the path to the Haar Cascade file
cascade_path = './haarcascade_frontalface_default.xml'

# Load the Haar Cascade file

video = cv2.VideoCapture(0)
face_detect = cv2.CascadeClassifier(cascade_path)

face_data = []
i=0
name = input('Enter your name: ')

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_detect.detectMultiScale(gray, 1.3, 5)
    
    for(x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50))
        
        if len(face_data) <= 100 and i % 10 == 0:
            face_data.append(resized_img)
        i+=1
        cv2.putText(frame, str(len(face_data)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
        
    cv2.imshow('frame', frame)
    k = cv2.waitKey(1)
    
    if len(face_data) == 50:
        break
    
video.release()
cv2.destroyAllWindows()


face_data = np.array(face_data)
face_data = face_data.reshape(100, -1)


# Check if 'names.pkl' exists and handle accordingly
if 'names.pkl' not in os.listdir(data_dir):
    names = [name] * 100
    with open(os.path.join(data_dir, 'names.pkl'), 'wb') as f:
        pickle.dump(names, f)
else:
    with open(os.path.join(data_dir, 'names.pkl'), 'rb') as f:
        names = pickle.load(f)
        names = names + [name] * 100
    
    with open(os.path.join(data_dir, 'names.pkl'), 'wb') as f:
        pickle.dump(names, f)

# Check if 'face_data.pkl' exists and handle accordingly
if 'face_data.pkl' not in os.listdir(data_dir):
    with open(os.path.join(data_dir, 'face_data.pkl'), 'wb') as f:
        pickle.dump(face_data, f)
else:
    with open(os.path.join(data_dir, 'face_data.pkl'), 'rb') as f:
        faces = pickle.load(f)
        faces = np.concatenate((faces, face_data), axis=0)
    
    with open(os.path.join(data_dir, 'face_data.pkl'), 'wb') as f:
        pickle.dump(faces, f) 