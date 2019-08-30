import os

list_of_numbers = sorted([i.split(".")[0] for i in os.listdir('images')], key=int)

list_of_names = ['images/' + i + '.jpeg' for i in list_of_numbers]
print(list_of_names)

for i in range(1, len(list_of_numbers)):
    os.rename(list_of_names[i], 'images/' + str(i) + '.jpeg')