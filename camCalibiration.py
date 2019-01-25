from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import cv2
import time

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

# Define the chess board rows and columns
chess_rows = 7
chess_cols = 5
block_size = 24 #24mm
time.sleep(0.1)
# Set the termination criteria for the corner sub-pixel algorithm
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, block_size, 0.001)

# Prepare the object points: (0,0,0), (1,0,0), (2,0,0), ..., (6,5,0). They are $
objectPoints = np.zeros((chess_rows * chess_cols, 3), np.float32)
objectPoints[:, :2] = np.mgrid[0:chess_rows, 0:chess_cols].T.reshape(-1, 2)

# Create the arrays to store the object points and the image points
objectPointsArray = []
imgPointsArray = []

ImgNumber = 0

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    #capture frame by frame
    image = frame.array
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (chess_rows, chess_cols), None)
    if ret:
        #refine the corner position
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        objectPointsArray.append(objectPoints)
        imgPointsArray.append(corners)
        cv2.drawChessboardCorners(gray, (chess_rows, chess_cols), corners, ret)
        ImgNumber += 1
        print("number of images= {}".format(ImgNumber))

        cv2.waitKey(2000) # 2 second to change the chess board pose
    cv2.imshow("output", gray)
    key = cv2.waitKey(1) & 0xFF
    rawCapture.truncate(0)
    #exit if 50 images are used	
    if ImgNumber == 50:
        break
    if key == ord("q"):
        break
# Calibrate the camera and save the results
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPointsArray, imgPointsArray, gray.shape[::-1], None, None)

# Print the camera Matrix
print(mtx)
# Print the camera calibration error
error = 0

for i in range(len(objectPointsArray)):
    imgPoints, _ = cv2.projectPoints(objectPointsArray[i], rvecs[i], tvecs[i], mtx, dist)
    error += cv2.norm(imgPointsArray[i], imgPoints, cv2.NORM_L2) / len(imgPoints)

print("Total error: ", error / len(objectPointsArray))

