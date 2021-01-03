#Usar
#python face_recognition_landmarks_01_image_without_cv2.py

import face_recognition
from PIL import Image, ImageDraw

def draw_face_landmarks(img, face_landmarks_list, landmarks_model):
  pil_image = Image.fromarray(img)
  d = ImageDraw.Draw(pil_image, 'RGBA')
  r=2 #radius ellipse
  for face_landmarks in face_landmarks_list:
    for face_part in face_landmarks:
      for x,y in face_landmarks[face_part]:
        d.ellipse((x-r, y-r, x+r, y+r), fill=(255,0,0), outline=(0,255,0))
  #pil_image.save("without.jpg")
  pil_image.show()

def find_face_landmarks(img, detection_model, landmarks_model):
  face_locations = face_recognition.face_locations(img, 0, detection_model)
  face_landmarks_list = face_recognition.face_landmarks(img, face_locations=face_locations, model=landmarks_model)
  draw_face_landmarks(img, face_landmarks_list, landmarks_model)

face_detection_model = 'cnn' #hog = faster | cnn = accurate
face_landmarks_model = 'large' #large = 68 points | small = 5 points

# Load an image
image = face_recognition.load_image_file("Buneco_Eko.jpg")
find_face_landmarks(image, face_detection_model, face_landmarks_model)

