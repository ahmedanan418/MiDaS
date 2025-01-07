import numpy as np
import cv2 as cv
import glob
import os
import pickle


# Get camera intrinsic parameters 
def calibration():
    chessboardSize = (7, 7)  # Update for 8x8 chessboard (inner corners)
    frameSize = (640, 480)

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points
    objp = np.zeros((1, chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:, :, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    images = glob.glob('captured_frames/*.jpg')  # Update path if necessary
    for image in images:
        img = cv.imread(image)
        if img is None:
            print(f"Failed to load image: {image}")
            continue
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)
        if ret:
            print(f"Chessboard detected in: {image}")
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(1000)
        else:
            print(f"Chessboard not detected in: {image}")

    cv.destroyAllWindows()

    if len(objpoints) == 0 or len(imgpoints) == 0:
        raise ValueError("No valid chessboard detections. Ensure proper images and chessboardSize.")

    ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)
    return cameraMatrix, ret, dist, rvecs, tvecs




def main():
    # Get camera intrinsic parameters
    cameraMatrix, ret, dist, rvecs, tvecs= calibration()


    # Save camera intrinsic parameters
    pickle.dump((cameraMatrix, dist), open( "calibration.pkl", "wb" ))
    pickle.dump(cameraMatrix, open( "cameraMatrix.pkl", "wb" ))
    pickle.dump(dist, open( "dist.pkl", "wb" ))


if __name__== "__main__":
    main()