import cv2
from scipy.spatial import distance
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Flatten, Input, Dense, Reshape
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from dlib import get_frontal_face_detector
import os
import argparse
from tqdm import tqdm

encoder = load_model('encoder_model.h5')
face_detector = get_frontal_face_detector()

def compare_faces(encodings, known_encodings, known_names, threshold = 51):
    names = []
    for i in range(len(encodings)):
        dists = [np.sum(np.square(encodings[i] - known_enc)) for known_enc in known_encodings]
        print(dists)
        print('Min dist: ' + str(min(dists)))
        if min(dists) > threshold:
            names.append('Unknown')
        else:
            names.append(known_names[i])
    return names

def get_embedding_list(faces):
    data = np.concatenate(faces, axis=0)
    data = np.expand_dims(data, -1)
    data = data / 255.
    return encoder.predict(data)

def get_boxes(image):
    detected_faces = face_detector(image, 1)
    face_frames = [(x.left(), x.top(),x.right(), x.bottom()) for x in detected_faces]
    return face_frames

def get_faces(image):
    faces = []
    detected_faces = face_detector(image, 1)
    face_frames = [(x.left(), x.top(),x.right(), x.bottom()) for x in detected_faces]
    for (left, top, right, bottom) in face_frames:
        img = image[top:bottom, left:right]
        img = cv2.resize(img, (64, 64))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces.append(np.expand_dims(img, 0))
        #cv2.rectangle(image, (left, top), (right, bottom), (255,0,0), 2)
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

def boxes_and_names(image, boxes, names):
    
    for i in range(len(boxes)):
        
        (left, top, right, bottom) = boxes[i]

        fr = int((bottom - top) / 4)

        cv2.line(image, (left, top), (left, top+fr), (0, 0, 255), 3)
        cv2.line(image, (left, bottom-fr), (left, bottom), (0, 0, 255), 3)
        cv2.line(image, (right, top), (right, top+fr), (0, 0, 255), 3)
        cv2.line(image, (right, bottom-fr), (right, bottom), (0, 0, 255), 3)
        cv2.line(image, (left, top), (left+fr, top), (0, 0, 255), 3)
        cv2.line(image, (right, top), (right-fr, top), (0, 0, 255), 3)
        cv2.line(image, (left, bottom), (left+fr, bottom), (0, 0, 255), 3)
        cv2.line(image, (right, bottom), (right-fr, bottom), (0, 0, 255), 3)

        cv2.putText(image, names[i], (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

def train_encoder():
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--folder", type=str,
        help="folder with faces dataset")
    ap.add_argument("-n", "--epochs", type=int,
        help="number of epochs to train", default=10)
    ap.add_argument("-p", "--path", type=str,
        help="path to save the model", default='encoder_model.h5')
    args = vars(ap.parse_args())

    folder = args["folder"]

    print('Creating input tensor')

    data = []
    for image in tqdm(os.listdir(folder)):
        img = cv2.imread(folder + '/' + image)
        img = cv2.resize(img,(64,64))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        data.append(np.expand_dims(img, 0))

    faces = np.concatenate(data, axis=0)
    faces = np.expand_dims(faces, -1)
    faces.shape
    print('Preparing data')
    faces = faces / 255.
    print('Shape of input tensor:', end=' ')
    print(faces.shape)

    # encoding
    input_ = Input((64, 64, 1)) # 64
    x = Conv2D(filters=8, kernel_size=2, strides=2, activation='relu')(input_) # 32
    x = Conv2D(filters=16, kernel_size=2, strides=2, activation='relu')(x) # 16
    x = Conv2D(filters=32, kernel_size=2, strides=2, activation='relu')(x) # 8
    x = Conv2D(filters=64, kernel_size=2, strides=2, activation='relu')(x) # 4
    x = Conv2D(filters=128, kernel_size=2, strides=2, activation='relu')(x) # 2
    flat = Flatten()(x)
    latent = Dense(128)(flat)

    # decoder
    reshape = Reshape((2,2,32)) #2
    conv_2t_1 = Conv2DTranspose(filters=128, kernel_size=2, strides=2, activation='relu') # 4
    conv_2t_2 = Conv2DTranspose(filters=64, kernel_size=2, strides=2, activation='relu') # 8
    conv_2t_3 = Conv2DTranspose(filters=32, kernel_size=2, strides=2, activation='relu') # 16
    conv_2t_4 = Conv2DTranspose(filters=16, kernel_size=2, strides=2, activation='relu') # 32
    conv_2t_5 = Conv2DTranspose(filters=1, kernel_size=2, strides=2, activation='sigmoid') # 64

    x = reshape(latent)
    x = conv_2t_1(x)
    x = conv_2t_2(x)
    x = conv_2t_3(x)
    x = conv_2t_4(x)
    decoded = conv_2t_5(x) # 64

    autoencoder = Model(input_, decoded)
    encoder = Model(input_, latent)

    decoder_input = Input((128,))
    x_ = reshape(decoder_input)
    x_ = conv_2t_1(x_)
    x_ = conv_2t_2(x_)
    x_ = conv_2t_3(x_)
    x_ = conv_2t_4(x_)
    decoded_ = conv_2t_5(x_) # 64
    decoder = Model(decoder_input, decoded_)

    print('Summary about model:')

    autoencoder.summary()    
    plot_model(autoencoder, to_file='model.png', show_shapes = True, show_layer_names = True)
    print('Pipeline of the model saved to model.png')
    print('================================')
    print('starting training')

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.fit(faces, faces, epochs = args['epochs'])

    print('Saving encoder model...')
    encoder.save(args['path'])