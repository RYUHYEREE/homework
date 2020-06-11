import ijson
import sys
import os
import cv2
import numpy as np

def debug(name, mat):
    cv2.imshow(name, cv2.resize(mat, (0,0), fx=0.5, fy=0.5))

split = 'train'  # 'val'  # specify which directory to convert
#split = 'val'

# Open JSON file. Because it huge file (1.5Gb) use ijson, scalable JSON parser.
seg_db = 'seg/'
src_db = 'drivable_lane_maps/'
tar_db = 'images/'

json_file = 'labels/bdd100k_labels_images_' + split + '.json'
f = open(json_file, 'rb')
images = ijson.items(f, 'item')

slow = True
#slow = False

i = 0
for image in images:
    i += 1
    print(i, image['name'])
    img_path = 'images/100k/' + split + '/' + image['name']
    seg_path = seg_db + 'color_labels/' + split + '/' + image['name'][:-4] + '_train_color.png'
    tar_path = tar_db + 'seg_lane/' + split + '/' + image['name'][:-4] + '.jpg'
    print(img_path)
    if slow:
        print(seg_path)
        print(tar_path)

    # if seg has certain image
    if not os.path.exists(seg_path):
        print("File not found in seg")
        print()
        continue
    print("Found", seg_path)

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    seg = cv2.imread(seg_path, cv2.IMREAD_COLOR)

    cv2.imwrite(tar_path, img)
    print('write to', tar_path)

    # -----txt.write(image['name']+'\n')
