import os
import cv2
import numpy as np
import open3d as o3d

rgb_dir = '/home/kota/dat/cnn/training/via-2.0.10/images/20200811_wooden/rgb'
depth_dir = '/home/kota/dat/cnn/training/via-2.0.10/images/20200811_wooden/depth'
left_dir = '/home/kota/dat/cnn/training/via-2.0.10/images/20200811_wooden/left'

# fx, fy, cx, cy are from self.M1 of `calibration_utils.py`
fx, fy = 857.81117191, 858.24022281
cx, cy = 639.20397033, 371.14222636
min_z = 1000.0
max_z = 10000.0  # don't show points further than 10m


def to_xyz(zz):
    xx, yy = np.meshgrid(np.arange(zz.shape[1]), np.arange(zz.shape[0]))
    xx = (xx - cx) * zz / fx
    yy = (yy - cy) * zz / fy
    xyz = np.dstack((xx, yy, zz))
    xyz = xyz.transpose(2, 0, 1).reshape(3, -1).transpose()

    xyz = xyz[xyz[:, 2] < max_z]

    return xyz


def visualize_points(points):
    # This works with open3d 0.5.0, but need changes with other versions
    # http://www.open3d.org/docs/release/tutorial/Basic/working_with_numpy.html
    pcd = o3d.PointCloud()
    pcd.points = o3d.Vector3dVector(xyz)

    o3d.visualization.draw_geometries([pcd])

    return


def show_depth(depth_map):
    depth_img = np.clip(depth_map, min_z, max_z)
    depth_img = ((depth_img - min_z) * 255.0 / (max_z - min_z))
    cv2.imshow("depth", depth_img.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyWindow("depth")


if __name__ == '__main__':
    depth_files = os.listdir(depth_dir)
    depth_files.sort()
    for depth_file in depth_files:
        zz = np.load(os.path.join(depth_dir, depth_file)).astype(np.float)  ## [mm]
        show_depth(zz)
        xyz = to_xyz(zz)
        visualize_points(xyz)

        break
