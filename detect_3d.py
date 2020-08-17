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

rect = (0,0,1,1)
rectangle = False
rect_over = False
def onmouse(event, x, y, flags, params):
    global rectangle,rect,ix,iy,roi

    # Draw Rectangle
    if event == cv2.EVENT_LBUTTONDOWN:
        rectangle = True
        ix,iy = x,y
        left_copy = left.copy()

    elif event == cv2.EVENT_MOUSEMOVE:
        if rectangle == True:
            left_copy = left.copy()
            cv2.rectangle(left_copy,(ix,iy),(x,y),(0,255,0),1)
            cv2.imshow('left', left_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        rectangle = False

        left_copy = left.copy()
        cv2.rectangle(left_copy,(ix,iy),(x,y),(0,255,0),1)

        rect = (min(ix,x),min(iy,y),abs(ix-x),abs(iy-y))
        cv2.imshow('left', left_copy)


def to_xyz(zz):
    xx, yy = np.meshgrid(np.arange(zz.shape[1]), np.arange(zz.shape[0]))
    xx = (xx - cx) * zz / fx
    yy = (yy - cy) * zz / fy
    xyz = np.dstack((xx, yy, zz))
    xyz = xyz.transpose(2, 0, 1).reshape(3, -1).transpose()

    xyz = xyz[xyz[:, 2] < max_z]

    return xyz


def visualize_points(xyz):
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

    return


def detect_pallet_3d(xyz):
    pcd = o3d.PointCloud()
    pcd.points = o3d.Vector3dVector(xyz)

    o3d.geometry.estimate_normals(cloud = pcd,
                                  search_param=o3d.KDTreeSearchParamHybrid(radius=50.0, max_nn=30))  # 50.0 -> 5.0cm
    normals = np.asarray(pcd.normals)

    v_vertical = np.array([0.0, -1.0, 0.0])

    print(xyz.shape)
    o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    depth_files = os.listdir(depth_dir)
    depth_files.sort()
    for depth_file in depth_files:
        left = cv2.imread(os.path.join(left_dir, depth_file.replace("depth", "left").replace(".npy", ".png")))
        cv2.imshow("left", left)
        cv2.setMouseCallback("left", onmouse)
        cv2.waitKey(0)
        cv2.destroyWindow("left")
        print("pallet rect: ", rect)

        zz = np.load(os.path.join(depth_dir, depth_file)).astype(np.float)  ## [mm]
        show_depth(zz)

        # xyz = to_xyz(zz)
        # visualize_points(xyz)

        xyz_roi = to_xyz(zz[rect[1]: rect[1] + rect[3],
                            rect[0]: rect[0] + rect[2]])
        visualize_points(xyz_roi)

        detect_pallet_3d(xyz_roi)

        break
