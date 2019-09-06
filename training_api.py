from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Flatten, Input, Dense, Reshape
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
import numpy as np
import cv2
from tqdm import tqdm
import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--folder", type=str,
	help="folder with faces dataset")
ap.add_argument("-n", "--epochs", type=int,
	help="number of epochs to train", default=10)
args = vars(ap.parse_args())

def train(folder=args['folder']):
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
    encoder.save('encoder_model.h5')

if __name__ == "__main__":
    train()