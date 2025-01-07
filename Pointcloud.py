import open3d as o3d
import numpy as np
import cv2 as cv
import pickle
import os


def load_calibration_data():
    # Load the camera calibration parameters
    try:
        cameraMatrix = pickle.load(open("cameraMatrix.pkl", "rb"))
        dist = pickle.load(open("dist.pkl", "rb"))
        return cameraMatrix, dist
    except FileNotFoundError as e:
        print(f"Calibration file not found: {e}")
        exit(1)
    except Exception as e:
        print(f"Error loading calibration data: {e}")
        exit(1)


def check_image_dimensions(rgb_img, depth_img):
    if rgb_img.shape[:2] != depth_img.shape[:2]:
        print("Error: RGB and Depth images must have the same dimensions.")
        exit(1)


def main():
    # Check if the RGB and depth image files exist
    rgb_path = "/home/ahmed/MiDaS/input/frame_0.jpg"
    depth_path = "/home/ahmed/MiDaS/output/frame_0-dpt_swin2_large_384.png"

    if not os.path.exists(rgb_path):
        print(f"RGB image not found at {rgb_path}")
        exit(1)
    if not os.path.exists(depth_path):
        print(f"Depth image not found at {depth_path}")
        exit(1)

    # Load depth and RGB images
    rgb_img = o3d.io.read_image(rgb_path)
    depth_img = o3d.io.read_image(depth_path)
    
    # Convert Open3D image to numpy arrays to access the dimensions
    rgb_np = np.asarray(rgb_img)
    depth_np = np.asarray(depth_img)
    
    # Debug: Print image details (size and channels)
    print(f"RGB Image: {rgb_np.shape[1]}x{rgb_np.shape[0]} with {rgb_np.shape[2]} channels")
    print(f"Depth Image: {depth_np.shape[1]}x{depth_np.shape[0]}")
    
    # Ensure depth is in the right format
    depth_np = depth_np.astype(np.float32)  # Ensure depth is float32
    
    # Check for NaN or Inf values in the depth image and replace them
    depth_np = np.nan_to_num(depth_np, nan=0.0, posinf=1000.0, neginf=0.0)
    
    # Convert back to Open3D image format
    depth_img = o3d.geometry.Image(depth_np)

    # Try creating the RGBD image with depth_scale = 1.0 (instead of 1000.0)
    try:
        rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_img, depth_img, depth_scale=1.0, convert_rgb_to_intensity=False
        )
        print("RGBD image created successfully")
    except Exception as e:
        print(f"Error creating RGBD image: {e}")
        exit(1)

    # Camera Intrinsics (replace with your actual camera matrix and dist)
    fx, fy, cx, cy = 1000, 1000, 640, 360
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(800, 800, fx, fy, cx, cy)

    # Testing with smaller image portion: Take a 100x100 region of the image for debugging
    rgb_np_small = rgb_np[:100, :100].copy()  # Ensure the array is contiguous
    depth_np_small = depth_np[:100, :100].copy()  # Ensure the array is contiguous

    # Convert the smaller images back to Open3D image format
    rgb_img_small = o3d.geometry.Image(rgb_np_small)
    depth_img_small = o3d.geometry.Image(depth_np_small)

    # Try creating the RGBD image with the small image
    try:
        rgbd_img_small = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_img_small, depth_img_small, depth_scale=1.0, convert_rgb_to_intensity=False
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img_small, intrinsic)
        print("Point cloud created successfully from small image")
    except Exception as e:
        print(f"Error generating point cloud from small image: {e}")
        exit(1)

    # Visualize point cloud
    try:# Example: Voxel downsampling with leaf size 0.1
        voxel_down_filter = o3d.geometry.VoxelDownFilter(voxel_size=0.1)
        pcd_downsampled = voxel_down_filter.filter_PointCloud(pcd)
        o3d.visualization.draw_geometries([pcd_downsampled])
        print("Point cloud visualized successfully")
    except Exception as e:
        print(f"Error visualizing point cloud: {e}")
        exit(1)

if __name__ == "__main__":
    main()