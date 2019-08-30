import cv2
import os
import numpy as np

i = 45000
while i<=50000:
    after = cv2.imread('images/' + str(i) + '.jpeg')
    before = cv2.imread('images/' + str(i-1) + '.jpeg')
    after = cv2.resize(after, (64, 64))
    before = cv2.resize(before, (64,64))

    after = after.tolist()
    before = before.tolist()

    if after == before:
        os.remove('images/' + str(i) + '.jpeg')
        print(str(i) + '.jpeg'+'was removed')
        i+=2
    else:
        i+=1
    print(i)