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

# proper calibration 4/2/2025
DIM=(920, 780)
K=np.array([[513.7644432897364, 0.0, 437.2970737078887], [0.0, 514.2705392759157, 396.11653395869695], [0.0, 0.0, 1.0]])
D=np.array([[-0.10719568512838897], [0.07046579419530903], [-0.11468531603999366], [0.06251065598507932]])

# wonkyness
# DIM=(1920, 1080)
# K=np.array([[803.6547764320203, 0.0, 955.0396150982891], [0.0, 796.5451741009057, 535.0841440675592], [0.0, 0.0, 1.0]])
# D=np.array([[-0.6531525767846975], [30.270081370092583], [-698.8900668252405], [-1780.698462938757]])


def main():
    cap = cv2.VideoCapture('/dev/video2')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    while True:
        result = handleFrame(cap)
        if not result or cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def handleFrame(cap: cv2.VideoCapture):
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        return False

    cv2.imshow("original", frame)

    # x_start, y_start, x_end, y_end = 550, 150, 1470, 930  # Adjust as needed
    # cropped_frame = frame[y_start:y_end, x_start:x_end]  # Crop the frame
    
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)    

    # cv2.imshow("cropped", cropped_frame)
    cv2.imshow("Undistorted", undistorted_frame)
    return True


if __name__ == '__main__':
    main()
