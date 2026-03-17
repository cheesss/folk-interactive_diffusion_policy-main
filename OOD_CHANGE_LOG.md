# OOD 변경 파일 정리

이 문서는 `folk-interactive_diffusion_policy-main` 안에서 OOD 모니터링과 LPB 스타일 dynamics 학습을 위해 추가/수정한 파일을 정리한 목록입니다.

## 실행 파일

- `rb10_eval_real_robot.py`
  - 저장소 원본 실행 파일입니다.
  - 기존 동작을 유지하도록 원복했습니다.

- `rb10_eval_real_robot_ood.py`
  - 원본 실행 파일과 분리된 OOD 전용 실행 엔트리포인트입니다.
  - `current_ood`, `predicted_ood` 계산, 실시간 시각화, obs key/shape 검사를 포함합니다.

## 런타임 모듈

- `diffusion_policy/real_world/ood_monitor.py`
  - expert latent bank와 dynamics checkpoint를 로드합니다.
  - 현재 관측의 `current_ood`, action sequence 기반의 `predicted_ood`를 계산합니다.
  - bank에 저장된 `shape_meta`와 현재 task의 `shape_meta`가 맞는지 검사합니다.

- `diffusion_policy/common/ood_utils.py`
  - base policy encoder를 이용해 obs를 latent로 바꾸는 유틸리티입니다.
  - nearest-neighbor 거리 계산과 OOD 점수 정규화 함수가 들어 있습니다.

## 학습 코드

- `diffusion_policy/model/ood/latent_dynamics_model.py`
  - LPB 아이디어를 따라 현재 latent/proprio와 미래 action chunk를 넣어 미래 latent를 예측하는 모델입니다.
  - temporal transformer 기반 predictor를 사용합니다.

- `diffusion_policy/workspace/train_ood_dynamics_workspace.py`
  - 기존 diffusion_policy workspace 패턴에 맞춘 OOD dynamics 학습 워크스페이스입니다.
  - frozen base policy encoder를 재사용해 latent regression을 학습합니다.

- `diffusion_policy/dataset/son_replay_ood_dynamics_dataset.py`
  - expert HDF5와 optional rollout HDF5를 읽어 OOD dynamics 학습 샘플로 변환합니다.
  - 현재 obs, future obs, action chunk를 하나의 샘플로 반환합니다.

## 오프라인 스크립트

- `diffusion_policy/scripts/export_ood_assets.py`
  - expert demo를 latent bank `.pt` 파일로 내보내는 스크립트입니다.
  - bank latent, 통계값, lowdim key, shape_meta를 함께 저장합니다.

- `diffusion_policy/scripts/replay_buffer_to_hdf5.py`
  - 실험 중 생성된 `replay_buffer.zarr`를 학습용 HDF5 포맷으로 바꾸는 변환 스크립트입니다.

## 설정 파일

- `diffusion_policy/config/son_train_ood_dynamics_real_workspace.yaml`
  - OOD dynamics 학습용 workspace 설정입니다.
  - optimizer, dataloader, epoch 수, transformer 하이퍼파라미터가 들어 있습니다.

- `diffusion_policy/config/son_export_ood_assets.yaml`
  - expert latent bank export용 설정입니다.

- `diffusion_policy/config/task/son_pick_and_place_image_ood.yaml`
  - OOD dynamics 학습에서 사용할 task/dataset shape_meta 설정입니다.

## 패키지 초기화

- `diffusion_policy/model/ood/__init__.py`
  - OOD 모델 패키지 초기화 파일입니다.

## 참고

- `expert_latent_bank.pt`
  - 훈련된 모델이 아니라 expert latent 기준집합입니다.
  - base policy encoder로 expert obs를 latent로 바꿔 저장한 결과물입니다.

- `OOD dynamics checkpoint`
  - 실제로 학습되는 모델입니다.
  - `predicted_ood` 계산에 사용됩니다.
