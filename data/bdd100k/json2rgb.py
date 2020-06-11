import ijson
import cv2
import numpy as np

split = 'train'  # 'val'  # specify which directory to convert

# Open JSON file. Because it huge file (1.5Gb) use ijson, scalable JSON parser.
filename = 'labels/bdd100k_labels_images_' + split + '.json'
f = open(filename, 'rb')

# --------I commented out the codes for generating image_file_name_list.txt
# --------txt = open('image_file_name_list.txt', 'w+')

# Parse json. Returns in iteratable format.
i = 0
images = ijson.items(f, 'item')

for image in images:
    i += 1
    print(i, image['name'])

    img = cv2.imread('images/100k/' + split + '/' + image['name'], cv2.IMREAD_COLOR)
    rgb = cv2.imread('drivable_maps/color_labels/' + split + '/' + image['name'][:-4] + '_drivable_color.png',
                     cv2.IMREAD_COLOR)

    for label in image['labels']:
        if label['category'] == 'lane':  # Match if it is lane. Put traffic light, and it parse traffic light.
            polygon = label['poly2d'][0]
            pts = np.array(polygon['vertices'], np.int32)  # get all vertices if labeled with lane
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(rgb, [pts], bool(polygon['closed']), (255, 0, 255), thickness=10)  # draw polylines
        elif label['category'] == 'traffic sign':
            polygon = np.fromiter(label['box2d'].values(), dtype=float)
            pts = np.array([[polygon[0], polygon[1]], [polygon[2], polygon[1]], [polygon[2], polygon[3]], [polygon[0], polygon[3]]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(rgb, [pts], True, (0, 255, 0), thickness=10)
        elif label['category'] == 'car':
            polygon = np.fromiter(label['box2d'].values(), dtype=float)
            pts = np.array([[polygon[0], polygon[1]], [polygon[2], polygon[1]], [polygon[2], polygon[3]], [polygon[0], polygon[3]]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(rgb, [pts], True, (255, 255, 255), thickness=10)
        elif label['category'] == 'person':
            polygon = np.fromiter(label['box2d'].values(), dtype=float)
            pts = np.array([[polygon[0], polygon[1]], [polygon[2], polygon[1]], [polygon[2], polygon[3]], [polygon[0], polygon[3]]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(rgb, [pts], True, (64, 64, 0), thickness=10)

    # Debug
    # cv2.imshow('rgb', rgb)
    blend = cv2.bitwise_or(img, rgb)
    # cv2.imshow('blend', blend)
    cv2.imwrite('drivable_lane_maps/color_labels/' + split + '/' + image['name'][0:-4] + '_drivable_color.png', rgb)
    # if cv2.waitKey(0) & 0xFF == ord('q'):  # wait for key input every conversion
    #     break

    # -----txt.write(image['name']+'\n')
