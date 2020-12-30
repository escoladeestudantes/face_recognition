#Usar
#python 03_recognize_images_opencv.py

import face_recognition
import pickle
import cv2
import numpy as np
import glob
import os


data = pickle.loads(open('brazza_napalm.pickle', "rb").read())

path = ''
for infile in glob.glob(os.path.join(path, '*.jpg')):
	# Load an image with an unknown face
	unknown_image = cv2.imread(infile)
	image = unknown_image[:, :, ::-1]

	# Find all the faces and face encodings in the unknown image
	face_locations = face_recognition.face_locations(image, 0, 'cnn')
	face_encodings = face_recognition.face_encodings(image, face_locations)

	# Loop through each face found in the unknown image
	for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
		# See if the face is a match for the known face(s)
		matches = face_recognition.compare_faces(data["encodings"], face_encoding)
		name = ""

		# If a match was found in known_face_encodings, just use the first one.
		#if True in matches:
		#    first_match_index = matches.index(True)
		#    name = data['names'][first_match_index]

		# Or instead, use the known face with the smallest distance to the new face
		face_distances = face_recognition.face_distance(data["encodings"], face_encoding)
		best_match_index = np.argmin(face_distances)
		if (matches[best_match_index] and face_distances[best_match_index] < 0.55):
		    name = data["names"][best_match_index]
		    print('{}: {}'.format(name, face_distances[best_match_index]))

		# Draw a box around the face using the Pillow module
		cv2.rectangle(unknown_image, (left, top), (right, bottom), (0, 255, 0), 7)

		# Draw a label with a name below the face
		cv2.rectangle(unknown_image, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
		font = cv2.FONT_HERSHEY_DUPLEX
		cv2.putText(unknown_image, name, (left + 6, bottom - 6), font, 0.9, (0, 0, 0), 2)

	cv2.imshow('Imagem', unknown_image)
	cv2.waitKey(0)
cv2.destroyAllWindows()

