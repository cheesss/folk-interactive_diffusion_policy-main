if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import argparse
import os
import sys
import h5py
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


def _ensure_rb10_import(rb10_root):
    rb10_root = os.path.abspath(rb10_root)
    if rb10_root not in sys.path:
        sys.path.insert(0, rb10_root)
    from rb.RB10 import RB10  # pylint: disable=import-error
    return RB10


def rotmat_to_6d(rotmats):
    rotmats = np.asarray(rotmats)
    if rotmats.ndim == 2:
        rotmats = rotmats[np.newaxis, :, :]
    r1 = rotmats[:, :, 0]
    r2 = rotmats[:, :, 1]
    return np.concatenate([r1, r2], axis=1)


def resize_images(image_list, size):
    return [cv2.resize(img, size) for img in image_list]


def compute_r(steps):
    if steps <= 0:
        return np.array([], dtype=np.float32)
    return -np.arange(steps - 1, -1, -1, dtype=np.float32)


def compute_bins(r_values, num_bins, r_min=None, r_max=None):
    if r_min is None:
        r_min = float(np.min(r_values))
    if r_max is None:
        r_max = float(np.max(r_values))
    if r_min == r_max:
        r_max = r_min + 1.0
    return np.linspace(r_min, r_max, num_bins + 1, dtype=np.float32)


def _get_obs_group(demo_group):
    if "observations" in demo_group:
        return demo_group["observations"]
    if "obs" in demo_group:
        return demo_group["obs"]
    raise RuntimeError("No observations/obs group found in demo.")


def process_demo(input_demo, output_demo, robot, resize_wh, r_key):
    output_obs = output_demo.create_group("obs")
    input_obs = _get_obs_group(input_demo)

    input_joint = np.array(input_obs["joint"])
    input_image_wrist = np.array(input_obs["image_wrist"])
    input_image_scene = np.array(input_obs["image_scene"])
    input_gripper = np.array(input_obs["gripper"])
    n_steps = len(input_joint)

    tcp_poses = []
    tcp_rotmats = []
    for i in range(n_steps):
        tcp_se3 = robot.fkine(input_joint[i])
        tcp_poses.append(tcp_se3.t)
        tcp_rotmats.append(tcp_se3.R)

    tcp_poses = np.array(tcp_poses)
    tcp_rotmats = np.array(tcp_rotmats)
    tcp_quats = R.from_matrix(tcp_rotmats).as_quat()
    tcp_quats = np.array([-q if q[3] < 0 else q for q in tcp_quats])
    tcp_rotation_6d = rotmat_to_6d(tcp_rotmats)

    output_image_wrist = resize_images(list(input_image_wrist), resize_wh)
    output_image_scene = resize_images(list(input_image_scene), resize_wh)
    output_image_wrist = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in output_image_wrist]
    output_image_scene = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in output_image_scene]
    output_image_wrist = np.array(output_image_wrist)
    output_image_scene = np.array(output_image_scene)

    output_obs.create_dataset("position", data=tcp_poses[:-1])
    output_obs.create_dataset("rotation_6d", data=tcp_rotation_6d[:-1])
    output_obs.create_dataset("quat", data=tcp_quats[:-1])
    output_obs.create_dataset("joint", data=input_joint[:-1])

    gripper = input_gripper[:-1]
    if gripper.ndim == 1:
        gripper = gripper[:, np.newaxis]
    output_obs.create_dataset("gripper", data=gripper)
    output_obs.create_dataset("image0", data=output_image_wrist[:-1])
    output_obs.create_dataset("image1", data=output_image_scene[:-1])

    next_pose = tcp_poses[1:]
    next_rotation_6d = tcp_rotation_6d[1:]
    next_gripper = input_gripper[1:]
    if next_gripper.ndim == 1:
        next_gripper = next_gripper[:, np.newaxis]
    actions = np.concatenate([next_pose, next_rotation_6d, next_gripper], axis=-1)
    output_demo.create_dataset("actions", data=actions.astype(np.float32))

    r = compute_r(len(output_obs["position"]))
    output_obs.create_dataset(r_key, data=r)
    return len(r)


def main():
    parser = argparse.ArgumentParser(description="Postprocess + r/value_bin pipeline")
    parser.add_argument("--input", nargs="+", required=True, help="Input HDF5 files")
    parser.add_argument("--output", required=True, help="Output HDF5 file")
    parser.add_argument("--rb10_root", default="/home/ws/Desktop/robotory_rb10_ros2/scripts")
    parser.add_argument("--resize_w", type=int, default=320)
    parser.add_argument("--resize_h", type=int, default=240)
    parser.add_argument("--r_key", default="r")
    parser.add_argument("--num_bins", type=int, default=201)
    parser.add_argument("--r_min", type=float, default=None)
    parser.add_argument("--r_max", type=float, default=None)
    parser.add_argument("--out_key", default="value_bin")
    args = parser.parse_args()

    RB10 = _ensure_rb10_import(args.rb10_root)
    robot = RB10()
    resize_wh = (args.resize_w, args.resize_h)

    output_demo_idx = 0
    all_r = []
    with h5py.File(args.output, "w") as output_file:
        output_data = output_file.create_group("data")
        for input_path in args.input:
            with h5py.File(input_path, "r") as input_file:
                input_data = input_file["data"]
                demo_names = sorted(
                    input_data.keys(),
                    key=lambda x: int(x.split("_")[1]) if "_" in x else 0
                )
                for demo_name in tqdm(demo_names, desc=f"Processing {os.path.basename(input_path)}"):
                    input_demo = input_data[demo_name]
                    output_demo = output_data.create_group(f"demo_{output_demo_idx}")
                    steps = process_demo(input_demo, output_demo, robot, resize_wh, args.r_key)
                    all_r.append(output_demo["obs"][args.r_key][:])
                    output_demo_idx += 1
                    print(f"Processed demo_{output_demo_idx - 1} ({steps} steps)")

        if not all_r:
            raise RuntimeError("No r values collected; check input files.")

        all_r = np.concatenate(all_r, axis=0).astype(np.float32)
        edges = compute_bins(all_r, args.num_bins, args.r_min, args.r_max)
        r_min = float(edges[0])
        r_max = float(edges[-1])
        print(f"Using r range [{r_min}, {r_max}] with {args.num_bins} bins")

        for demo_key in output_data.keys():
            obs = output_data[demo_key]["obs"]
            r = obs[args.r_key][:].astype(np.float32)
            bin_idx = np.digitize(r, edges) - 1
            bin_idx = np.clip(bin_idx, 0, args.num_bins - 1).astype(np.int64)
            if args.out_key in obs:
                del obs[args.out_key]
            obs.create_dataset(args.out_key, data=bin_idx)

    print(f"Saved output to {args.output}")


if __name__ == "__main__":
    main()
