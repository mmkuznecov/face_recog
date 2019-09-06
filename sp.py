from shutil import copyfile
import os

for i in range(1, 1001):
	src = 'faces_imgs/' + str(i) + '.jpeg'
	dst = 'fcs/' + str(i) + '.jpeg'
	copyfile(src, dst)
