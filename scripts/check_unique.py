import cv2
import os
from tqdm import tqdm

data = []

for image in tqdm(os.listdir('faces_imgs')):
    img = cv2.imread('faces_imgs/' + image)
    img = cv2.resize(img, (64, 64))
    img = img.tolist()
    data.append(img)

lol = 0

for im in tqdm(data):
    if data.count(im) > 1:
        print('repeated')
        lol += 1

print(lol)