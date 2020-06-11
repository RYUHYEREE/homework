import ijson
import cv2
import numpy as np

split = 'train'
filename = 'labels/bdd100k_labels_images_'+split+'.json'
f = open(filename, 'rb')
images = ijson.items(f, 'item')
i = 0
for image in images:
    i += 1
    print(i, image['name'])

    img = cv2.imread('images/100k/'+split+'/'+image['name'], cv2.IMREAD_COLOR)
    cv2.imshow('img', cv2.resize(img, (640, 320)))

    rgb = cv2.imread('drivable_lane_maps/color_labels/'+split+'/'+image['name'][:-4]+'_drivable_color.png', cv2.IMREAD_COLOR)
    cv2.imshow('rgb', cv2.resize(rgb, (640, 320)))

    lbl = cv2.imread('drivable_lane_maps/labels/'+split+'/'+image['name'][:-4]+'_drivable_id.png', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('lbl', cv2.resize(lbl, (640, 320)))

    blend = cv2.bitwise_or(img, rgb)
    cv2.imshow('blend', cv2.resize(blend, (640, 320)))

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
