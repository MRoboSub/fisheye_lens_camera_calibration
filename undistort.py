import cv2

import numpy as np
import os
import glob

import sys

# You should replace these 3 lines with the output in calibration step
# DIM=(920, 780)
# DIM=(1920, 1080)
# K=np.array([[519.6730777316743, 0.0, 494.5258514004607], [0.0, 519.0897449631934, 370.841559431988], [0.0, 0.0, 1.0]])
# D=np.array([[-0.11572710555486544], [0.05335040836382976], [-0.0782535398440889], [0.04633489792326648]])
DIM=(920, 780)
K=np.array([[513.7644432897364,               0.0, 437.29707370788870], 
            [              0.0, 514.2705392759157, 396.11653395869695], 
            [              0.0,               0.0,               1.0]])
D=np.array([[-0.10719568512838897], 
            [ 0.07046579419530903], 
            [-0.11468531603999366], 
            [ 0.06251065598507932]])

def undistort(img_path):    
    img = cv2.imread(img_path)
    h,w = img.shape[:2]    
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)    
    cv2.imshow("original", img)
    cv2.imshow("undistorted", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    for p in sys.argv[1:]:
        undistort(p)
