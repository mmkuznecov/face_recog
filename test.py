from face_recog import *
import cv2

known_encodings, known_names = get_known_encodings('faces')
print(known_names[0])
print(known_encodings[0])