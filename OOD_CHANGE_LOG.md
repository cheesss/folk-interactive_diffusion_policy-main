# OOD and Recovery Implementation Map

This document identifies the project-specific code added to the Diffusion Policy base for RB10 OOD monitoring and recovery-aware policy-bridge experiments.

## Runtime entry point

### `rb10_eval_real_robot_ood.py`

Real-robot inference entry point kept separate from the upstream rollout script. It adds:

- current and predicted OOD scoring
- real-time diagnostic visualization
- observation-key and tensor-shape validation
- hooks for recovery guidance during policy inference

`rb10_eval_real_robot.py` remains the reference task-policy execution path.

## Runtime monitoring

### `diffusion_policy/real_world/ood_monitor.py`

- loads an expert latent bank and latent-dynamics checkpoint
- calculates representation distance for the current observation
- predicts future latent drift under a candidate action sequence
- verifies compatibility between bank metadata and the runtime policy

### `diffusion_policy/common/ood_utils.py`

- encodes observations with the frozen task-policy encoder
- computes nearest-neighbor latent distance
- normalizes raw distances into OOD diagnostic scores

## Latent dynamics

### `diffusion_policy/model/ood/latent_dynamics_model.py`

Predicts a future latent representation from the current latent state, proprioception, and an action horizon. The current public path uses a temporal transformer predictor.

### `diffusion_policy/workspace/train_ood_dynamics_workspace.py`

Training workspace following the repository's existing Hydra workspace pattern. It keeps the task-policy encoder frozen while optimizing future-latent regression.

### `diffusion_policy/dataset/son_replay_ood_dynamics_dataset.py`

Converts expert and optional rollout HDF5 files into tuples containing current observations, future observations, and action chunks for dynamics training.

## Asset preparation

### `diffusion_policy/scripts/export_ood_assets.py`

Encodes expert demonstrations with the task-policy encoder and exports:

- reference latents
- normalization statistics
- observation metadata
- policy horizon and representation settings

### `diffusion_policy/scripts/replay_buffer_to_hdf5.py`

Converts collected replay-buffer data into the HDF5 format expected by the dynamics-training pipeline.

## Configuration

### `diffusion_policy/config/son_train_ood_dynamics_real_workspace.yaml`

Defines the latent-dynamics model, optimizer, data loaders, training schedule, and output paths.

### `diffusion_policy/config/son_export_ood_assets.yaml`

Defines the reference policy, expert dataset, latent-bank output, and device used during asset export.

### `diffusion_policy/config/task/son_pick_and_place_image_ood.yaml`

Defines the RB10 task observations, actions, image inputs, and dataset metadata for the OOD research path.

## Generated research assets

The following assets are intentionally not committed:

- `expert_latent_bank.pt`: expert observation support encoded by the task policy
- OOD dynamics checkpoints: learned future-latent predictors
- task-policy and LPB checkpoints
- expert demonstration and robot rollout datasets

These artifacts may contain unpublished experimental data or machine-specific paths. Contact the author for research access.
