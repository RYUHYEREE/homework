import ijson
import sys
import os
import cv2
import numpy as np

def debug(name, mat):
    cv2.imshow(name, cv2.resize(mat, (0,0), fx=0.5, fy=0.5))

#split = 'train'  # 'val'  # specify which directory to convert
split = 'val'

# Open JSON file. Because it huge file (1.5Gb) use ijson, scalable JSON parser.
seg_db = 'seg/'
src_db = 'drivable_lane_maps/'
tar_db = 'drivable_lane_seg_maps/'

class_bgr = [
    [142, 0, 0], # car 
    [0, 220, 220], #  
]

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
    rgb_path = src_db + 'color_labels/' + split + '/' + image['name'][:-4] + '_drivable_color.png'
    seg_path = seg_db + 'color_labels/' + split + '/' + image['name'][:-4] + '_train_color.png'
    tar_path = tar_db + 'color_labels/' + split + '/' + image['name'][:-4] + '_train_color.png'
    print(img_path)
    if slow:
        print(rgb_path)
        print(seg_path)
        print(tar_path)

    # if seg has certain image
    if not os.path.exists(seg_path):
        print("File not found in seg")
        print()
        continue
    print("Found", seg_path)

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    seg = cv2.imread(seg_path, cv2.IMREAD_COLOR)
    new = rgb.copy() 

    
    for bgr in class_bgr:
        
        # Extracting object
        th = np.array(bgr)
        mask = cv2.inRange(seg, th, th)
        cla = cv2.bitwise_and(seg, seg, mask=mask)

        # Foreground addtion
        new = cv2.add(new, cla)

        if slow:
            print("bgr", bgr)
            debug('mask', mask)
            debug('cla', cla) # debug foreground of class object one by one
            cv2.waitKey(0)

    if slow:
        debug('img', img)
        debug('rgb', rgb)
        debug('seg', seg)
        debug('new', new)

        blend = cv2.bitwise_or(img, new)
        debug('blend', blend)
    cv2.imwrite(tar_path, new)
    print('write to', tar_path)
    print()
    if cv2.waitKey(0) & 0xFF == ord('q'):  # wait for key input every conversion
        break

    # -----txt.write(image['name']+'\n')
