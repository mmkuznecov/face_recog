import cv2
import urllib3
import numpy as np
import time
import os

i = max(list(map(lambda x: int(x.split('.')[0]), os.listdir("./faces_imgs")))) + 1
while True:
    try:
        http = urllib3.PoolManager()
        req = http.request('GET', 'https://thispersondoesnotexist.com/image', preload_content=False)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)
        img = cv2.resize(img, (128, 128))

        cv2.imwrite('.faces_imgs/' + str(i)+'.jpeg', img)
        i += 1
        print('image was added ' + str(i - 1))
        time.sleep(0.9)
        '''if i - 1 == 100000:
            break'''
    except Exception as e:
        print('problem with connection')
        print(e)
