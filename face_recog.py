import cv2
from scipy.spatial import distance
import numpy as np
from keras.models import load_model
from dlib import get_frontal_face_detector
import os

encoder = load_model('models/encoder_model.h5')
face_detector = get_frontal_face_detector()

def compare_faces(encodings, known_encodings, known_names, threshold = 40):
    names = []
    for i in range(len(encodings)):
        dists = [np.sum(np.square(encodings[i] - known_enc)) for known_enc in known_encodings]
        dist = min(dists)
        if dist > threshold:
            names.append('Unknown')
        else:
            names.append(known_names[i])
    return names

def get_embedding_list(faces):
    data = np.concatenate(faces, axis=0)
    data = np.expand_dims(data, -1)
    data = data / 255.
    return encoder.predict(data)

def get_faces(image):
    faces = []
    detected_faces = face_detector(image, 1)
    face_frames = [(x.left(), x.top(),x.right(), x.bottom()) for x in detected_faces]
    for (left, top, right, bottom) in face_frames:
        img = image[top:bottom, left:right]
        img = cv2.resize(img, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces.append(np.expand_dims(img, 0))
    return faces

def get_known_encodings(folder):
    encodings = []
    names = []
    for image in os.listdir(folder):
        face = cv2.imread(folder + '/' + image)
        if len(get_faces(face)) > 1:
            print('Image ' + image + ' contains more than one face')
        elif len(get_faces(face)) == 0:
            print('Image ' + image + ' does not contain faces')
        else:
            print('Extracting face encodings from ' + image)
            encodings.append(get_embedding_list(get_faces(face))[0])
            names.append(image.split('.')[0])
    return (encodings, names)