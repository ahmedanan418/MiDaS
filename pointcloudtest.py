import open3d as o3d
import numpy as np
import os
import pickle

def load_calibration_data():
    cameraMatrix = pickle.load(open("cameraMatrix.pkl", "rb"))
    dist = pickle.load(open("dist.pkl", "rb"))
    return cameraMatrix, dist

# Load RGB and depth images
rgb_img = o3d.io.read_image("/home/ahmed/MiDaS/input/frame_0.jpg")
depth_img = o3d.io.read_image("/home/ahmed/MiDaS/output/frame_0-dpt_swin2_large_384.png")

# Convert RGB to uint8 and depth to float32
rgb_img_np = np.asarray(rgb_img)
rgb_img = o3d.geometry.Image(rgb_img_np.astype(np.uint8))

depth_img_np = np.asarray(depth_img, dtype=np.float32)

# Normalize depth to meters
print("Depth Image Before Scaling - Min:", np.min(depth_img_np), "Max:", np.max(depth_img_np))
depth_scale = 5.0 / 255.0  # Adjust scale based on your data
depth_img_np *= depth_scale
print("Depth Image After Scaling - Min:", np.min(depth_img_np), "Max:", np.max(depth_img_np))
depth_img_np[depth_img_np <= 0] = 1e-3  # Replace invalid values
depth_img = o3d.geometry.Image(depth_img_np)

# Load camera calibration parameters
cameraMatrix, dist = load_calibration_data()

# Convert cameraMatrix to Open3D intrinsic
fx, fy = cameraMatrix[0, 0], cameraMatrix[1, 1]
cx, cy = cameraMatrix[0, 2], cameraMatrix[1, 2]
width, height = np.asarray(rgb_img).shape[1], np.asarray(rgb_img).shape[0]

intrinsic = o3d.camera.PinholeCameraIntrinsic()
intrinsic.set_intrinsics(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)

# Create RGBD Image
try:
    rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_img, depth_img, convert_rgb_to_intensity=False
    )
    print("RGBD Image created successfully.")
except Exception as e:
    print("Error creating RGBD Image:", e)
    exit()

# Generate Point Cloud
try:
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img, intrinsic)
    
    try:
        # Save the point cloud to a file
        output_dir = "Pointclouds"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = os.path.join(output_dir, "point_cloud.ply")   # You can change the filename and format as needed
        o3d.io.write_point_cloud(output_file, pcd)
        print(f"Point cloud saved successfully to {output_file}.")
    
    except Exception as e:
        print("Cannot save pointcloud", e)
    
    # Visualize pointcloud
    o3d.visualization.draw_geometries([pcd])
except Exception as e:
    print("Error creating Point Cloud:", e)
