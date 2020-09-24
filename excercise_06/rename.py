import os
import shutil

data_dir = './data/bird/'
file_list = os.listdir(data_dir)
print(file_list)

for i, filename in enumerate(file_list):
    original = os.path.join(data_dir, filename)
    # renamed = os.path.join(data_dir, 'bird.{}{}'.format(i, os.path.splitext(original)[-1]))
    # print(original, renamed)
    # os.rename(original, renamed)
    if i == 60:
        break

    if i < 30:
        print('bird/{},0'.format(filename))
    else:
        print('bird/{},1'.format(filename))
