from dlib import get_frontal_face_detector
import numpy as np
import cv2
import os
from tqdm import tqdm

face_detector = get_frontal_face_detector()

faces = 0
no_faces = 0
too_many_faces = 0
fcs = len(os.listdir('images/'))

known_faces = os.listdir('croped_faces')

for img in tqdm(os.listdir('images')):
    
    if img not in known_faces:

        im = cv2.imread('images/' + img)
        im = cv2.resize(im, (256, 256))
        detected_faces = face_detector(im, 1)
        if len(detected_faces) == 0:
            no_faces += 1
        elif len(detected_faces) == 1:
            faces += 1

            (left, top, right, bottom) = (detected_faces[0].left(), detected_faces[0].top(), detected_faces[0].right(), detected_faces[0].bottom())
            im = im[top:bottom, left:right]
            im = cv2.resize(im, (64, 64))
            im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            cv2.imwrite('croped_faces/' + img, im)
        else:
            too_many_faces += 1

print('No faces found on: ' + str(no_faces / fcs * 100) + '%')
print('Ok on: ' + str(faces / fcs * 100) + '%')
print('Too many faces found on: ' + str(too_many_faces / fcs * 100) + '%')