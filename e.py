import numpy as np
import cv2
import os
from tqdm import tqdm
from dlib import get_frontal_face_detector

face_detector = get_frontal_face_detector()

data = []
i = 0
for image in tqdm(os.listdir('croped_faces')):

    img = cv2.imread('croped_faces/' + image)
    print(img)
    print(image)
    try:
        print(img.shape)
    except:
        i+= 1
        im = cv2.imread('images/' + image)
        im = cv2.resize(im, (256, 256))
        detected_faces = face_detector(im, 1)
        (left, top, right, bottom) = (detected_faces[0].left(), detected_faces[0].top(), detected_faces[0].right(), detected_faces[0].bottom())
        im = im[top:bottom, left:right]
        im = cv2.resize(im, (64, 64))
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        cv2.imwrite('croped_faces/' + image, im)
    data.append(np.expand_dims(img, 0))

print(i)