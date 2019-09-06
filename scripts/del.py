import cv2
import numpy as np
import os

a = cv2.imread('images/24547.jpeg')
b = cv2.imread('images/24548.jpeg')

c = cv2.resize(a, (128, 128))
d = cv2.resize(b, (128, 128))

e = cv2.resize(a, (64, 64))
f = cv2.resize(b, (64, 64))

a = a.tolist()
b = b.tolist()
c = c.tolist()
d = d.tolist()
e = e.tolist()
f = f.tolist()

if a == b:
    print('true')
    #os.remove('images/27549.jpeg')
if c == d:
    print('true')

if e == f:
    print('true')