import cv2
import os
from os import walk
import sys

import numpy as np
from pathlib import Path


def get_bricket_coords(img, cam_num, corr_h=0, corr_w=0):
    roi_img, roi_coords = get_roi_from_img(img, cam_num)
    roi_gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(roi_gray, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)

    bricket_contour = sorted(contours, key=lambda x: cv2.contourArea(x),
                             reverse=True)[0]

    mask = np.zeros_like(img)
    cv2.drawContours(mask, [bricket_contour], 0, 255, -1)
    out = np.zeros_like(img)
    out[mask == 255] = img[mask == 255]

    (y, x, _) = np.where(mask == 255)
    (miny, minx) = (np.min(y), np.min(x))
    (maxy, maxx) = (np.max(y), np.max(x))

    roi_min_point, _ = roi_coords
    point_min = (miny + corr_h + roi_min_point[0],
                 minx + corr_w + roi_min_point[1])
    point_max = (maxy - corr_h + roi_min_point[0],
                 maxx - corr_w + roi_min_point[1])

    return point_min, point_max


def get_roi_from_img(img, cam_num):
    roi_coords = {
        'cam1': ((0, 430), (1080, 1735)),
        'cam2': ((0, 100), (1080, 1780)),
        'cam3': ((0, 620), (1080, 1780)),
        'cam4': ((), ()),
        'cam5': ((), ()),
        'cam6': ((0, 420), (1080, 1780)),
    }
    point_top, point_bottom = roi_coords[cam_num]
    min_y, min_x = point_top
    max_y, max_x = point_bottom
    return img[min_y:max_y, min_x:max_x], roi_coords[cam_num]


def delete_file_duplicates(root_directory, changed_suffix, orig_suffix):
    direc = Path(root_directory)
    all_files_with_crops = sorted(list(direc.rglob('*' + changed_suffix)))
    #   all_files_with_crops = folder.rglob('*' + changed_suffix)
    removed = 0
    for file in all_files_with_crops:
        os.remove(file)
        removed += 1

    print(f'Deleted {removed} file duplicates')


# root_directory = 'data'
# crop_directory = 'data_crop'
# camera_number = 'cam6'
#
# types = ('*.jpeg', '*.jpg', '*.png')
# orig_suffix = '_Changed_Orig.png'
# changed_suffix = '_Changed.jpg'
#
# processed = 0
#
# for subdir, dirs, files in os.walk(root_directory + '/' + camera_number):
#     for folder in dirs:
#         p = root_directory + '/' + camera_number + '/' + folder + '/'
#
#         delete_file_duplicates(p, changed_suffix, orig_suffix)
#
#         _, _, images_list = next(walk(p))
#
#         path = crop_directory + '/' + camera_number + '/' + folder
#         Path(path).mkdir(parents=True, exist_ok=True)
#
#         for image in images_list:
#             img = cv2.imread(p + image, cv2.IMREAD_COLOR)
#
#             point_min, point_max = get_bricket_coords(img, camera_number, corr_h=0, corr_w=0)
#             img = img[point_min[0]:point_max[0], point_min[1]:point_max[1]]
#
#             cv2.imwrite(path + '/' + image, img)
#             processed += 1
#             if processed % 100 == 0:
#                 print('Processed ' + str(processed) + ' images...')
