#Usar
#python face_recognition_landmarks_01_image.py

import face_recognition
import cv2

def draw_face_landmarks(img, face_landmarks_list, landmarks_model):
  for face_landmarks in face_landmarks_list:
    for face_part in face_landmarks:
      for x,y in face_landmarks[face_part]:
        img = cv2.circle(img, (x,y), 3, (0,255,0), -1)
        img = cv2.circle(img, (x,y), 2, (0,0,255), -1)
  cv2.imshow('Imagem', img)
  cv2.waitKey(0)

def find_face_landmarks(img, detection_model, landmarks_model):
  face_locations = face_recognition.face_locations(img, 0, detection_model)
  face_landmarks_list = face_recognition.face_landmarks(img, face_locations=face_locations, model=landmarks_model)
  draw_face_landmarks(img, face_landmarks_list, landmarks_model)

face_detection_model = 'cnn' #hog = faster | cnn = accurate
face_landmarks_model = 'small' #large = 68 points | small = 5 points

# Load an image with an unknown face
image = cv2.imread("Buneco_Eko.jpg")
#image = cv2.imread("CesarMC.jpg")
find_face_landmarks(image, face_detection_model, face_landmarks_model)


