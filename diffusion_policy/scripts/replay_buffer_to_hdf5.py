"""실험 중 저장한 replay_buffer.zarr를 학습용 HDF5 형식으로 바꾸는 변환 스크립트."""

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import argparse
import pathlib

import h5py
import numpy as np

from diffusion_policy.common.replay_buffer import ReplayBuffer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-zarr", required=True, help="Path to replay_buffer.zarr")
    parser.add_argument("--output-hdf5", required=True, help="Output HDF5 path")
    args = parser.parse_args()

    buffer = ReplayBuffer.create_from_path(args.input_zarr, mode="r")
    output_path = pathlib.Path(args.output_hdf5)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as file:
        data_group = file.create_group("data")
        for episode_idx in range(buffer.n_episodes):
            episode = buffer.get_episode(episode_idx, copy=True)
            demo_group = data_group.create_group(f"demo_{episode_idx}")
            obs_group = demo_group.create_group("obs")

            for key, value in episode.items():
                if key == "action":
                    demo_group.create_dataset("actions", data=np.asarray(value))
                elif key in ("timestamp", "stage"):
                    demo_group.create_dataset(key, data=np.asarray(value))
                else:
                    obs_group.create_dataset(key, data=np.asarray(value))

    print(f"Saved {buffer.n_episodes} episodes to {output_path}")


if __name__ == "__main__":
    main()
