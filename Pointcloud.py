import open3d as o3d
import numpy as np
import cv2 as cv
import glob

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
    images = glob.glob('/images/*.png') # Edit path everytime you try
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




# Load depth and RGB images
rgb_img= o3d.io.read_image("/home/ahmed/MiDaS/input/")  # Edit path everytime you try
depth_img= o3d.io.read_image("/home/ahmed/MiDaS/output/")   # Edit path everytime you try
# RGBD image generation
rgbd_img= o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_img,depth_img)

# Get camera intrinsic parameters
cameraMatrix, ret, dist, rvecs, tvecs= calibration()

# Convert cameraMatrix to Open3D intrinsic
fx, fy = cameraMatrix[0, 0], cameraMatrix[1, 1]
cx, cy = cameraMatrix[0, 2], cameraMatrix[1, 2]
intrinsic = o3d.camera.PinholeCameraIntrinsic()
intrinsic.set_intrinsics(width=640, height=480, fx=fx, fy=fy, cx=cx, cy=cy)

# Generate Point Cloud
pcd= o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img,cameraMatrix)

# Transform point cloud to align with the original depth camera frame
pcd.transform([[1, 0, 0, 0],[0, -1, 0, 0],[0, 0, -1, 0],[0, 0, 0, 1]])

# Save or visualize the point cloud
o3d.io.write_point_cloud("output.ply", pcd)
o3d.visualization.draw_geometries([pcd])
