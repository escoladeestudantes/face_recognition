#Usar
#python face_recognition_landmarks_02_video.py

import face_recognition
import cv2

def draw_face_landmarks(img, face_landmarks_list, landmarks_model):
  for face_landmarks in face_landmarks_list:
    for face_part in face_landmarks:
      for x,y in face_landmarks[face_part]:
        img = cv2.circle(img, (x,y), 3, (0,255,0), -1)
        img = cv2.circle(img, (x,y), 2, (0,0,255), -1)
  cv2.imshow('Frame', img)

def find_face_landmarks(img, detection_model, landmarks_model):
  face_locations = face_recognition.face_locations(img, 0, detection_model)
  face_landmarks_list = face_recognition.face_landmarks(img, face_locations=face_locations, model=landmarks_model)
  draw_face_landmarks(img, face_landmarks_list, landmarks_model)

face_detection_model = 'cnn' #hog = faster | cnn = accurate
face_landmarks_model = 'large' #large = 68 points | small = 5 points

#cap = cv2.VideoCapture('EduardoMarinho01.mp4')
cap = cv2.VideoCapture('FabioBrazza01.mp4')
ret=True

while(ret == True):
  ret, frame = cap.read()
  if (ret==True):
    find_face_landmarks(frame, face_detection_model, face_landmarks_model)
    if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
        break
cap.release()
cv2.destroyAllWindows()


