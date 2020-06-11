import ijson
import cv2
import numpy as np
from PIL import Image
import os
from os import walk

split = 'train'
images = []

for (dirpath, dirnames, filenames) in walk('./drivable_lane_seg_maps/color_labels/' + split):
    images.extend([i.split('_', 1)[0] for i in filenames])
    break

print(images)
i = 0
for image in images:
    i += 1
    print(i, image)

    # rgb : 256 bit color labeled ground truth
    rgb = cv2.imread('drivable_lane_seg_maps/color_labels/'+split+'/'+image+'_train_color.png', cv2.IMREAD_COLOR)

    # ref
    ref_path = 'drivable_maps/labels/'+split+'/'+image+'_drivable_id.png'
    ref = np.array(Image.open(ref_path))

    mask = { 
        (0, 0, 0): 0, # 0 black background
        (0, 0, 255): 1, # 1 red directly drivable area
        (255, 0, 0): 2, # 2 blue alternatively drivable area
        (255, 0, 255): 3, # 3 pink lane
        (60, 20, 220) : 4, # 4 person
        (142, 0, 0) : 5, # 5  car
        (100, 60, 0) : 6, # 6 bus
        (90, 0, 0) : 7, # 7 caravan
        (230, 0, 0) : 8, # 8 motorcycle
        (70, 0, 0) : 9, # 9 trucK
    }

    lbl = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.int8) # new array of the same size
    for k in mask:
        lbl[(rgb == k).all(axis=2)] = mask[k]   # put mask's postfix, if all rgb matches with mask's prefix
                                                # 2 if (255, 0, 0) mathces ...

    cv2.imwrite('drivable_lane_seg_maps/labels/'+split+'/'+image+'_drivable_id.png', lbl)
    
    # Debug
    # print('rgb', rgb) # [0, 0, 0], [0, 0, 255], etc
    # print('rgb shape', rgb.shape) # (720,  1280, 3)
    # print('rgb unique', np.unique(rgb))
    # cv2.imshow('rgb', rgb)

    #print('ref', ref)
    #print('ref shape', ref.shape)
    #print('ref unique', np.unique(ref))
    # cv2.imshow('ref', ref)

    #print('lbl', lbl)
    #print('lbl shape', lbl.shape)
    #print('lbl unique', np.unique(lbl))
    # cv2.imshow('lbl', lbl)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

