import cv2
import time
import numpy as np
from utils import find_mask_contour_area

# Creating a VideoCapture object
cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)

# Delay to wait for camera
time.sleep(3)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Set threshold
    lower_green = (25, 50, 25)
    upper_green = (120, 255, 255)

    mask = cv2.inRange(hsv, lower_green, upper_green)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    area = find_mask_contour_area(frame, mask)
    print(area)

    cv2.imshow("frame", result)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
