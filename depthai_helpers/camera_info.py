import yaml
import numpy as np


def write_camera_info(
    file_name,
    camera_name,
    image_shape,
    camera_matrix,
    distortion,
    projection_matrix=None,
    rectification_matrix=None,
    extrinsics=None,
):
    if not file_name.endswith(".yaml"):
        print("{} is not a yaml file.".format(file_name))

    if projection_matrix is None:
        projection_matrix = np.zeros((3, 4))
        projection_matrix[:, :3] = camera_matrix.copy()
        if extrinsics is not None:
            projection_matrix[:, 3] = extrinsics["t"]

    if rectification_matrix is None:
        rectification_matrix = np.eye(3)

    info = {}
    info["camera_name"] = camera_name
    info["image_width"] = image_shape[0]
    info["image_height"] = image_shape[1]

    info["camera_matrix"] = {
        "rows": 3,
        "cols": 3,
        "data": [float(x) for x in camera_matrix.flatten()],
    }

    info["distortion_model"] = "plumb_bob"
    info["distortion_coefficients"] = {
        "rows": 1,
        "cols": len(distortion.flatten()),
        "data": [float(x) for x in distortion.flatten()],
    }

    info["rectification_matrix"] = {
        "rows": 3,
        "cols": 3,
        "data": [float(x) for x in rectification_matrix.flatten()],
    }

    info["projection_matrix"] = {
        "rows": 4,
        "cols": 3,
        "data": [float(x) for x in projection_matrix.flatten()],
    }

    # with open(file_name, "w") as f:
    #     f.write(yaml.dump(info, default_flow_style=False))
    with open(file_name, "w") as f:
        yaml.dump(info, f, default_flow_style=None)


if __name__ == "__main__":
    intrinsics_file = "intrinsics.npz"
    extrinsics_file = "extrinsics.npz"

    intrinsics = np.load(intrinsics_file)
    extrinsics = np.load(extrinsics_file)

    write_camera_info(
        "camera_info.yaml", "left", (1280, 720), intrinsics["M1"], intrinsics["D1"]
    )
