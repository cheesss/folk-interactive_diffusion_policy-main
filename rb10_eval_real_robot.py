#!/home/vision/anaconda3/envs/robodiff/bin/python

# 실행코드
# python rb10_eval_real_robot.py --input data/outputs/19.53.16_train_diffusion_unet_hybrid_son_pick_and_place_image/checkpoints/epoch\=0900-train_loss\=0.000.ckpt --output data/results
"""
Usage:
(robodiff)$ python eval_real_robot.py -i <ckpt_path> -o <save_dir> --robot_ip <ip_of_ur5>

================ Human in control ==============
Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start evaluation (hand control over to policy).
Press "Q" to exit program.

================ Policy in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly! 

Recording control:
Press "S" to stop evaluation and gain control back.
"""

# %%
import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import torch
import dill
import hydra
import pathlib
import skvideo.io
from omegaconf import OmegaConf
import scipy.spatial.transform as st
from diffusion_policy.real_world.rb10_real_env import RealEnv   # 새로 만듬
# from diffusion_policy.real_world.spacemouse_shared_memory import Spacemouse
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.real_inference_util import (
    get_real_obs_resolution, 
    get_real_obs_dict)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.cv2_util import get_image_transform


OmegaConf.register_new_resolver("eval", eval, replace=True)

@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')   # checkpoint
@click.option('--output', '-o', required=True, help='Directory to save recording')   
@click.option('--robot_ip', '-ri', default="192.168.111.50", required=True, help="UR5's IP address e.g. 192.168.0.204")
@click.option('--match_dataset', '-m', default=None, help='Dataset used to overlay and adjust initial condition')   
@click.option('--match_episode', '-me', default=None, type=int, help='Match specific episode from the match dataset')
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--steps_per_inference', '-si', default=6, type=int, help="Action horizon for inference.")   # 몇개의 action 실행할건지
@click.option('--max_duration', '-md', default=60, help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")   # 20Hz ??
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
def main(input, output, robot_ip, match_dataset, match_episode,
    vis_camera_idx, init_joints, 
    steps_per_inference, max_duration,
    frequency, command_latency):
    

    # load checkpoint; checkpoint의 cfg 및 파라미터들 다 가져옴
    ckpt_path = input
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']   # yaml에 있던 변수들 설정값
    cls = hydra.utils.get_class(cfg._target_)   # WorkSpace 설정
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    # 여기서 workspace.model에 cfg.policy가 들어감


    # hacks for method-specific setup.
    action_offset = 0
    delta_action = False  
    if 'diffusion' in cfg.name:
        # diffusion model
        policy: BaseImagePolicy
        policy = workspace.model   # state_dicts의 model을 가져옴 (가중치 값들)
        if cfg.training.use_ema:
            policy = workspace.ema_model   # ema_model 가져옴 (가중치 값들)

        device = torch.device('cuda')
        policy.eval().to(device)

        # set inference params
        policy.num_inference_steps = 16 # DDIM inference iterations; 노이즈 제거 step 수
        policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1   # 과거부터 horizon 뽑고, obs만큼 빼고, 1 더하기

    else:
        raise RuntimeError("Unsupported policy type: ", cfg.name)


    # setup experiment
    dt = 1/frequency

    obs_res = get_real_obs_resolution(cfg.task.shape_meta)   # obs의 image 해상도 (width, height)
    n_obs_steps = cfg.n_obs_steps   # obs 관측 step 수
    print("n_obs_steps: ", n_obs_steps)   # obs 관측 step수 (2)
    print("steps_per_inference:", steps_per_inference)   # 예측한 action sequence에서 몇개의 action 실행할건지 (6)
    print("action_offset:", action_offset)   # action 지연 실행 (0)


    # sharedmemory에 데이터들 쌓기; 같은 공유 공간 사용
    with SharedMemoryManager() as shm_manager:
        with RealEnv(
            output_dir=output, 
            robot_ip=robot_ip, 
            frequency=frequency,   
            n_obs_steps=n_obs_steps,   
            obs_image_resolution=obs_res,   # (84, 84)
            obs_float32=True,   
            init_joints=init_joints,   # False
            enable_multi_cam_vis=True,   # 실시간 시각화 
            record_raw_video=False,   # 영상 저장 
            # number of threads per camera view for video recording (H.264)
            thread_per_video=3,
            # video recording quality, lower is better (but slower).
            video_crf=21,
            shm_manager=shm_manager) as env:
            cv2.setNumThreads(1)


            # Realsense-viewer에서 설정
            # Should be the same as demo
            # realsense exposure
            # env.realsense.set_exposure(exposure=120, gain=0)
            # realsense white balance
            # env.realsense.set_white_balance(white_balance=5900)

            print("Waiting for realsense")
            time.sleep(1.0)

            print("Warming up policy inference")
            
            # obs 받아오기
            obs = env.get_obs()

            with torch.no_grad():
                policy.reset()

                # 받은 obs에서 image 정규화 및 다듬기, pose 다듬기
                obs_dict_np = get_real_obs_dict(
                    env_obs=obs, shape_meta=cfg.task.shape_meta)
                
                # shape_meta 계층구조는 유지하면서 np --> tensor로 변환, 텐서 배치차원 추가
                obs_dict = dict_apply(obs_dict_np, 
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                
                # obs로 action 예측
                result = policy.predict_action(obs_dict)   # {'action': ~ , 'action_pred': ~}
                # 실제 실행할 action trajectory
                action = result['action'][0].detach().to('cpu').numpy()   # [0]은 배치차원 제거, tensor --> np
                assert action.shape[-1] == 10   # action 차원: 3 pos + 6 rot + 1 gripper
                del result

            print('Ready!')
            while True:
                
                # ========== policy control loop ==============
                try:
                    # start episode
                    policy.reset()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay   # 시스템시간, 영상 로그용
                    t_start = time.monotonic() + start_delay   # 로봇 제어 시간

                    env.start_episode(eval_t_start)   # 영상 저장 시작
                    # wait for 1/30 sec to get the closest frame actually
                    # reduces overall latency; 카메라 프레임 잘 받아오도록
                    frame_latency = 1/30
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")
                    iter_idx = 0   # trajectory 실행 개수
                    term_area_start_timestamp = float('inf')
                    perv_target_pose = None
                    while True:
                        # calculate timing; 실행할 action 만큼 기다릴 시간
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                        print('get_obs')
                        obs = env.get_obs()
                        obs_timestamps = obs['timestamp']
                        print(f'Obs latency {time.time() - obs_timestamps[-1]}')

                        # run inference; action 예측
                        with torch.no_grad():
                            s = time.time()
                            obs_dict_np = get_real_obs_dict(
                                env_obs=obs, shape_meta=cfg.task.shape_meta)
                            obs_dict = dict_apply(obs_dict_np, 
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                            # action 예측 
                            result = policy.predict_action(obs_dict)
                            # this action starts from the first obs step
                            action = result['action'][0].detach().to('cpu').numpy()   # 실행할 action[Horizon, Action_Dim]
                            print('Inference latency:', time.time() - s)
                       
                        # convert policy action to env actions
                        if delta_action:   # False
                            assert len(action) == 1
                            if perv_target_pose is None:
                                perv_target_pose = obs['robot_eef_pose'][-1]
                            this_target_pose = perv_target_pose.copy()
                            this_target_pose[[0,1]] += action[-1]
                            perv_target_pose = this_target_pose
                            this_target_poses = np.expand_dims(this_target_pose, axis=0)

                        else:   # len(action): Horizon / len(target_pose): 9
                            this_target_poses = np.zeros((len(action), action.shape[-1]), dtype=np.float64)
                            # this_target_poses[:] = target_pose
                            # this_target_poses[:,[0,1]] = action   
                            this_target_poses[:, :action.shape[-1]] = action

                        # deal with timing
                        # the same step actions are always the target for
                        action_timestamps = (np.arange(len(action), dtype=np.float64) + action_offset
                            ) * dt + obs_timestamps[-1]
                        action_exec_latency = 0.01
                        curr_time = time.time()
                        is_new = action_timestamps > (curr_time + action_exec_latency)   # 현재시점 이후 action만 실행
                        
                        if np.sum(is_new) == 0:   # 전부 지나버림
                            # exceeded time budget, still do something
                            this_target_poses = this_target_poses[[-1]]   # 마지막 action이라도 실행
                            # schedule on next available step
                            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                            action_timestamp = eval_t_start + (next_step_idx) * dt
                            print('Over budget', action_timestamp - curr_time)
                            action_timestamps = np.array([action_timestamp])

                        else:   # is_new = 1 인것만 실행
                            this_target_poses = this_target_poses[is_new]
                            action_timestamps = action_timestamps[is_new]

                        # clip actions; 범위 바꿈
                        # this_target_poses[:,:2] = np.clip(
                        #     this_target_poses[:,:2], [0.25, -0.45], [0.77, 0.40])
                        # delta action이라 clip 안함!!!!!!!!!!!!!!!!!!!!!!!!!!
                        # this_target_poses[:,:3] = np.clip(
                        #     this_target_poses[:,:3], [-0.50, -0.90, 0.095], [0.40, -0.37, 0.81])
                        
                        # execute actions; 실제 action 실행부분; 
                        env.exec_actions(
                            actions=this_target_poses,
                            timestamps=action_timestamps
                        )
                        print(f"Submitted {len(this_target_poses)} steps of actions.")


                        # 's' 누르면 종료
                        key_stroke = cv2.pollKey()
                        if key_stroke == ord('s'):
                            # Stop episode
                            # Hand control back to human
                            env.end_episode()
                            print('Stopped.')
                            break


                        # auto termination; 한계시간 지나면 종료
                        terminate = False
                        if time.monotonic() - t_start > max_duration:
                            terminate = True
                            print('Terminated by the timeout!')

                        if terminate:
                            env.end_episode()
                            break


                        # wait for execution; 로봇이 action 여러개 실행할동안 기다림
                        precise_wait(t_cycle_end - frame_latency)
                        iter_idx += steps_per_inference

                except KeyboardInterrupt:
                    print("Interrupted!")
                    # stop robot.
                    env.end_episode()
                
                print("Stopped.")



# %%
if __name__ == '__main__':
    main()
