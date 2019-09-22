from face_recog_lib import *
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('fa.jpg')

known_encodings, known_names = get_known_encodings('faces')

boxes = get_boxes(image)
print(len(boxes))

faces = get_faces(image)

print('Extracting faces')

encodings = get_embedding_list(faces)

print('Extracting embeddings')

names = compare_faces(encodings, known_encodings, known_names)

print('Getting names')
print(known_names)

boxes_and_names(image, boxes, names)

print('Drawing boxes')

cv2.imwrite('output.jpg', image)