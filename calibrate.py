import numpy as np
import cv2 as cv
import glob
import os
import pickle


# Get camera intrinsic parameters 
def calibration():
    chessboardSize = (9, 6)
    frameSize = (640, 480)
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((1, chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:, :, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    images = glob.glob('/captured_frames/*.jpg') # Edit path everytime you try
    for image in images:
        img = cv.imread(image)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)
        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(1000)
    cv.destroyAllWindows()

    ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

    return cameraMatrix, ret, dist, rvecs, tvecs



def main():
    
    # Capture frames 
    output_dir = "captured_frames"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(0)
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Webcam', frame)

        key = cv2.waitKey(1) & 0xFF  # Store the key press

        if key == ord('q'):
            break
        elif key == ord('f'): 
            frame_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            print(f"Captured and saved: {frame_path}")
            frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


    # Get camera intrinsic parameters
    cameraMatrix, ret, dist, rvecs, tvecs= calibration()


    # Save camera intrinsic parameters
    pickle.dump((cameraMatrix, dist), open( "calibration.pkl", "wb" ))
    pickle.dump(cameraMatrix, open( "cameraMatrix.pkl", "wb" ))
    pickle.dump(dist, open( "dist.pkl", "wb" ))


if __name__== "__main__":
    main()