"""
Usage:
Training:
python train.py --config-name=son_train_diffusion_transformer_real_hybrid_workspace task=son_push_image_abs
"""

import sys

# 버퍼링 없이 한줄마다 출력 (모아서 출력X)
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
# 수식을 사용할수있게함
OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    # 경로 : ~/diffusion_policy/diffusion_policy/config  <-- 여기서 yaml config 데이터 가져오나봄
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    # cfg에 있는 __target__ Class를 cls로 가져옴
    cls = hydra.utils.get_class(cfg._target_)
    # 가져온 class로 workspace 인스턴스 생성 후, run 실행
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
