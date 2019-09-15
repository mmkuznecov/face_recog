import os
for f in os.listdir('names'):
	os.mkdir(f.split('.')[0])
