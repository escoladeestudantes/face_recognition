#Usar
#python 05_recognize_video_fast.py

import face_recognition
import cv2
import numpy as np
import pickle

data = pickle.loads(open('brazza_napalm.pickle', "rb").read())

# Get a reference to webcam #0 (the default one)
#video_capture = cv2.VideoCapture('AtentadoNapalm_CypherRespect1.mp4')
video_capture = cv2.VideoCapture('AtentadoNapalm_Tokyo.mp4')

x_resize = 0.5
y_resize = 0.5

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]
    rgb_frame_small = cv2.resize(rgb_frame, (int(x_resize*rgb_frame.shape[1]), int(y_resize*rgb_frame.shape[0])), interpolation = cv2.INTER_AREA)
    # Find all the faces and face enqcodings in the frame of video
    face_locations = face_recognition.face_locations(rgb_frame_small, 0, 'cnn')
    face_encodings = face_recognition.face_encodings(rgb_frame_small, face_locations)

    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(data["encodings"], face_encoding)

        name = "Unknown"

        # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(data["encodings"], face_encoding)
        best_match_index = np.argmin(face_distances)
        if (matches[best_match_index] and face_distances[best_match_index] < 0.55):
            name = data["names"][best_match_index]
            print('{}: {}'.format(name, face_distances[best_match_index]))

        left = int(left/x_resize)
        right = int(right/x_resize)
        top = int(top/y_resize)
        bottom = int(bottom/y_resize)

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
