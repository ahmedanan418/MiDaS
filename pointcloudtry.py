import open3d as o3d
import pickle


def load_calibration_data():
    # Load the camera calibration parameters
    cameraMatrix = pickle.load(open("cameraMatrix.pkl", "rb"))
    dist = pickle.load(open("dist.pkl", "rb"))
    
    return cameraMatrix, dist




# Load depth and RGB images
rgb_img= o3d.io.read_image("/home/ahmed/MiDaS/input/frame_0.jpg")  # Edit path everytime you try
depth_img= o3d.io.read_image("/home/ahmed/MiDaS/output/frame_0-dpt_swin2_large_384.png")   # Edit path everytime you try

# RGBD image generation
rgbd_img= o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_img,depth_img)

# Load camera intrinsic parameters saved
cameraMatrix, dist= load_calibration_data()
print("Camera matrix:")
print(cameraMatrix)

# Convert cameraMatrix to Open3D intrinsic
fx, fy = cameraMatrix[0, 0], cameraMatrix[1, 1]
cx, cy = cameraMatrix[0, 2], cameraMatrix[1, 2]
print(fx )
print(fy )
print(cx )
print(cy)
intrinsic = o3d.camera.PinholeCameraIntrinsic()
intrinsic.set_intrinsics(width=640, height=480, fx=fx, fy=fy, cx=cx, cy=cy)
print(intrinsic)
# Generate Point Cloud
pcd= o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img,intrinsic)

# Transform point cloud to align with the original depth camera frame
pcd.transform([[1, 0, 0, 0],[0, -1, 0, 0],[0, 0, -1, 0],[0, 0, 0, 1]])

# Save or visualize the point cloud
o3d.io.write_point_cloud("output.ply", pcd)
o3d.visualization.draw_geometries([pcd])