

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import Int32, Float64, String, Float32MultiArray
from sensor_msgs.msg import JointState
import threading
import spatialmath.base as smb
from diffusion_policy.rb10_api.cobot import * 
from diffusion_policy.rb import *
from scipy.spatial.transform import Rotation as R

import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import scipy.interpolate as si
import scipy.spatial.transform as st
import numpy as np

# from rtde_control import RTDEControlInterface
# from rtde_receive import RTDEReceiveInterface

from diffusion_policy.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from diffusion_policy.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator

# 추가
def gripper_callback(msg):
    global latest_gripper_qpos
    latest_gripper_qpos = [msg.gGWD]
    # print("[DEBUG] gripper callback received:", latest_gripper_qpos[0])


def rot6d_to_rotvec(rot6d: np.ndarray) -> np.ndarray:
   
    a1 = rot6d[:3]
    a2 = rot6d[3:]

    b1 = a1 / np.linalg.norm(a1)

    a2_proj = np.dot(b1, a2) * b1
    b2 = a2 - a2_proj
    b2 = b2 / np.linalg.norm(b2)

    b3 = np.cross(b1, b2)

    R_mat = np.stack((b1, b2, b3), axis=1)  

    rot = R.from_matrix(R_mat)
    rot_vec = rot.as_rotvec()  

    return rot_vec
    

def se3_to_pos_rotvec(T: SE3):
    """
    T : spatialmath.SE3 or 4x4 homogeneous np.array
    return: (pos[3], rotvec[3])  # rotvec in rad
    """

    if isinstance(T, SE3):
        M = T.A  # 4x4 numpy array
    else:
        M = np.asarray(T)
        assert M.shape == (4,4), "T must be 4x4"

    pos = M[:3, 3].copy()
    rotvec = R.from_matrix(M[:3, :3]).as_rotvec()
    return np.hstack((pos, rotvec))
    

# current_joint: rad / target_pose: m, rad
def servoJ_rb(robot, current_joint, target_pose, dt, acc_pos_limit=40.0, acc_rot_limit=5.0):   # target_pose : rot_vec
    
    current_pose = robot.fkine(current_joint)   # SE3

    pos     = np.array(target_pose[:3])   # m
    rot_vec = np.array(target_pose[3:])   # rad         
    rotm    = R.from_rotvec(rot_vec).as_matrix()   
   
    T = np.eye(4)
    T[:3, :3] = rotm
    T[:3,  3] = pos
    target_pose = SE3(T)                   
    
    current_pose_rotvec = R.from_matrix(current_pose.R).as_rotvec()
  
    pose_error = current_pose.inv() * target_pose

    err_pos = target_pose.t - current_pose.t
    err_rot_ee = smb.tr2rpy(pose_error.R, unit='rad')
    err_rot_base = current_pose.R @ err_rot_ee
    err_6d = np.concatenate((err_pos, err_rot_base))

    J = robot.jacob0(current_joint)
    dq = np.linalg.pinv(J) @ err_6d

    if np.linalg.norm(dq[:3]) > acc_pos_limit:
        dq[:3] *= acc_pos_limit / np.linalg.norm(dq[:3])
    
    if np.linalg.norm(dq[3:]) > acc_rot_limit:
        dq[3:] *= acc_rot_limit / np.linalg.norm(dq[3:])
    
    next_joint = current_joint + dq * 1.0
    ServoJ(next_joint * 180 / np.pi, time1=dt)


def ServoJ(joint_deg, time1=0.002, time2=0.1, gain=0.005, lpf_gain=0.1):
    msg = f"move_servo_j(jnt[{','.join(f'{j:.3f}' for j in joint_deg)}],{time1},{time2},{gain},{lpf_gain})\n"
    SendCOMMAND(msg, CMD_TYPE.MOVE)
    

def ServoL(pose, time1=0.002, time2=0.1, gain=0.005, lpf_gain=0.1):
    msg = f"move_servo_l(pnt[{','.join(f'{p:.3f}' for p in pose)}],{time1},{time2},{gain},{lpf_gain})\n"
    SendCOMMAND(msg, CMD_TYPE.MOVE)


class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2


class RTDENode(Node):
    """ROS2 Node for RTDE controller to handle subscriptions"""
    def __init__(self, controller):
        super().__init__('rtde_interpolation_controller')
        self.controller = controller
        self.reentrant_group = ReentrantCallbackGroup()
        
        # Create subscribers for joint state and gripper position
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10,
            callback_group=self.reentrant_group
        )
        
        self.gripper_pos_sub = self.create_subscription(
            Int32,
            '/gripper/present_position',
            self.gripper_pos_callback,
            10,
            callback_group=self.reentrant_group
        )

        self.filtered_distance_sub = self.create_subscription(
            Float32MultiArray,
            '/filtered_distance',
            self.filtered_distance_callback,
            10,
            callback_group=self.reentrant_group
        )

        self.gripper_pub = self.create_publisher(Int32, '/gripper/command', 10)

    def gripper_control(self, target_gripper_qpos):
        msg = Int32()
        msg.data = int(target_gripper_qpos[0])
        self.gripper_pub.publish(msg)
    
    def joint_state_callback(self, msg):
        """Callback for joint state subscription"""
        with self.controller.joint_state_lock:
            if len(msg.position) >= 6:
                # Store joint positions in radians
                self.controller.joint_state = np.array(msg.position[:6])
    
    def gripper_pos_callback(self, msg):
        """Callback for gripper position subscription (Int32 message)"""
        with self.controller.gripper_pos_lock:
            self.controller.gripper_pos = msg.data
    
    def filtered_distance_callback(self, msg):
        """Callback for filtered distance subscription (Float32MultiArray message)"""
        with self.controller.filtered_distance_lock:
            # Convert Float32MultiArray to numpy array
            self.controller.filtered_distance = np.array(msg.data, dtype=np.float32)


class RTDEInterpolationController(mp.Process):
    """
    To ensure sending command to the robot with predictable latency
    this controller need its separate process (due to python GIL)
    """


    def __init__(self,
            shm_manager: SharedMemoryManager, 
            robot_ip, 
            frequency=125, 
            lookahead_time=0.1, 
            gain=300,
            max_pos_speed=0.25, # 5% of max speed
            max_rot_speed=0.16, # 5% of max speed
            launch_timeout=3,
            tcp_offset_pose=None,
            payload_mass=None,
            payload_cog=None,
            joints_init=None,
            joints_init_speed=1.05,
            soft_real_time=False,
            verbose=False,
            receive_keys=None,
            get_max_k=128,   # 30
            safety_threshold=0.05,  # meters, halt robot if filtered_distance < threshold
            ):
        """
        frequency: CB2=125, UR3e=500
        lookahead_time: [0.03, 0.2]s smoothens the trajectory with this lookahead time
        gain: [100, 2000] proportional gain for following target position
        max_pos_speed: m/s
        max_rot_speed: rad/s
        tcp_offset_pose: 6d pose
        payload_mass: float
        payload_cog: 3d position, center of gravity
        soft_real_time: enables round-robin scheduling and real-time priority
            requires running scripts/rtprio_setup.sh before hand.

        """
        # verify
        assert 0 < frequency <= 500
        assert 0.03 <= lookahead_time <= 0.2
        assert 100 <= gain <= 2000
        assert 0 < max_pos_speed
        assert 0 < max_rot_speed
        if tcp_offset_pose is not None:   # None
            tcp_offset_pose = np.array(tcp_offset_pose)
            assert tcp_offset_pose.shape == (6,)
        if payload_mass is not None:   # None
            assert 0 <= payload_mass <= 5
        if payload_cog is not None:   # None
            payload_cog = np.array(payload_cog)
            assert payload_cog.shape == (3,)
            assert payload_mass is not None
        if joints_init is not None:   # None
            joints_init = np.array(joints_init)
            assert joints_init.shape == (6,)

        super().__init__(name="RTDEPositionalController") 
        self.robot_ip = robot_ip
        self.frequency = frequency
        self.lookahead_time = lookahead_time
        self.gain = gain
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.launch_timeout = launch_timeout
        self.tcp_offset_pose = tcp_offset_pose
        self.payload_mass = payload_mass
        self.payload_cog = payload_cog
        self.joints_init = joints_init
        self.joints_init_speed = joints_init_speed
        self.soft_real_time = soft_real_time
        self.verbose = verbose
        self.safety_threshold = safety_threshold

        # build input queue; 예시 구조로부터 Queue 생성
        example = {
            'cmd': Command.SERVOL.value,
            # 'target_pose': np.zeros((6,), dtype=np.float64),
            'target_pose': np.zeros((10,), dtype=np.float64),  # 3 pos + 6 rot + 1 gripper
            'duration': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(   # 액션 담아놓을 메모리
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )

        # build ring buffer
        # if receive_keys is None:
        #     receive_keys = [   # 여기에 내가 받을 로봇 Pose로 바꾸기
        #         'ActualTCPPose',   # GetCurrentTCP
        #         'ActualTCPSpeed',   # 없음
        #         'ActualQ',   # GetCurrentJoint
        #         'ActualQd',   # 없음

        #         'TargetTCPPose',
        #         'TargetTCPSpeed',
        #         'TargetQ',
        #         'TargetQd'
        #     ]
        if receive_keys is None:
            receive_keys = [
                'robot_eef_pos',   
                'robot_eef_quat',
                'robot_gripper_qpos'
            ]

        # rtde_r = RTDEReceiveInterface(hostname=robot_ip)   
        # ToCB(ip=robot_ip)
        # rb10 = RB10()   # 여기서는 굳이 필요없을듯

        example = dict()
        
        # for key in receive_keys:           
        #     example[key] = np.array(getattr(rtde_r, 'get'+key)())   
        for key in receive_keys:
            if key == 'robot_eef_pos':
                example[key] = np.zeros((3,), dtype=np.float64)
            elif key == 'robot_eef_quat':
                example[key] = np.zeros((4,), dtype=np.float64)
            elif key == 'robot_gripper_qpos':
                example[key] = np.zeros((1,), dtype=np.float64)

        example['robot_receive_timestamp'] = time.time()
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(   # state 담아놓을 메모리
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()  
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys

        self.gripper_pub = None   # 그리퍼 제어용 퍼블리셔
        
        # ROS subscription state variables (used in run() process)
        self.joint_state = None
        self.gripper_pos = None
        self.filtered_distance = None
        self.joint_state_lock = threading.Lock()
        self.gripper_pos_lock = threading.Lock()
        self.filtered_distance_lock = threading.Lock()

    # ========= launch method ===========
    def start(self, wait=True):   
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[RTDEPositionalController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.STOP.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()
    
    def stop_wait(self):  
        self.join()
    
    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        
    # ========= command methods ============
    def servoL(self, pose, duration=0.1):   # 안씀
        """
        duration: desired time to reach pose
        """
        assert self.is_alive()
        assert(duration >= (1/self.frequency))
        pose = np.array(pose)
        assert pose.shape == (6,)   

        message = {
            'cmd': Command.SERVOL.value,
            'target_pose': pose,
            'duration': duration
        }
        self.input_queue.put(message)

    def schedule_waypoint(self, pose, target_time):   # 이거 사용
        assert target_time > time.time()
        pose = np.array(pose)
        # assert pose.shape == (6,)
        assert pose.shape == (10,)  # 3 pos + 6 rot + 1 gripper   

        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'target_time': target_time
        }
        self.input_queue.put(message)

    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()
    
    # ========= main loop in process ============
    def run(self):
        # enable soft real-time
        if self.soft_real_time:
            os.sched_setscheduler(
                0, os.SCHED_RR, os.sched_param(20))

        # start rtde
        robot_ip = self.robot_ip

        # rtde_c = RTDEControlInterface(hostname=robot_ip)   # 바꾸기!
        # rtde_r = RTDEReceiveInterface(hostname=robot_ip)   # 바꾸기!
        ToCB(ip=robot_ip)
        rb10 = RB10()
        CobotInit()

        # Real or Simulation
        SetProgramMode(PG_MODE.REAL)
        # SetProgramMode(PG_MODE.SIMULATION)


        # Initialize ROS2
        rclpy.init(args=None)
        
        # Create ROS2 node for subscriptions
        self.ros2_node = RTDENode(self)
        
        # Create executor to process ROS2 callbacks in separate thread (similar to dataset_gen.py)
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(self.ros2_node)
        
        # Run executor in a separate thread to process callbacks at full rate
        executor_thread = threading.Thread(target=executor.spin, daemon=True)
        executor_thread.start()
        
        # Store references for cleanup
        self.executor = executor
        self.executor_thread = executor_thread  

        try:
            if self.verbose:   # False
                print(f"[RTDEPositionalController] Connect to robot: {robot_ip}")

            # set parameters
            # if self.tcp_offset_pose is not None:   # None
            #     rtde_c.setTcp(self.tcp_offset_pose)
            # if self.payload_mass is not None:   # None
            #     if self.payload_cog is not None:
            #         assert rtde_c.setPayload(self.payload_mass, self.payload_cog)
            #     else:
            #         assert rtde_c.setPayload(self.payload_mass)
            
            # init pose
            if self.joints_init is not None:   # None
                # assert rtde_c.moveJ(self.joints_init, self.joints_init_speed, 1.4)
                MoveJ(self.joints_init, self.joints_init_speed, 1.4)

            # main loop
            dt = 1. / self.frequency

            # Wait for initial joint state and gripper position from ROS2 topics
            time.sleep(0.1)  # Give ROS2 some time to receive messages
            with self.joint_state_lock:
                if self.joint_state is None:
                    # Fallback to API if no ROS message received yet
                    j = GetCurrentJoint()
                    current_joint = np.array([j.j0, j.j1, j.j2, j.j3, j.j4, j.j5]) * np.pi / 180   # rad
                else:
                    current_joint = self.joint_state.copy()
            
            # Get current gripper position from subscription
            with self.gripper_pos_lock:
                if self.gripper_pos is not None:
                    current_gripper = float(self.gripper_pos)
                else:
                    current_gripper = 2100  # Fallback to open state if not received yet
            
            curr_se3 = rb10.fkine(current_joint)   # m, rad (SE3)
            curr_pose = se3_to_pos_rotvec(curr_se3)   
            
            # use monotonic time to make sure the control loop never go backward
            curr_t = time.monotonic()
            # Initialize last_waypoint_time in virtual time (starts at 0 pause duration)
            last_waypoint_time = curr_t
            # Initialize with current pose + current gripper position (7D: 3 pos + 3 rotvec + 1 gripper)
            curr_pose_with_gripper = np.concatenate([curr_pose, [current_gripper]])
            pose_interp = PoseTrajectoryInterpolator(   # abs
                times=np.array([curr_t]),     # [ time ]
                poses=np.array([curr_pose_with_gripper])   # [ [x,y,z,rx,ry,rz,gripper] ]
            )

            iter_idx = 0
            keep_running = True

            t_start = time.monotonic()   # 수동 제어 주기 맞추기
            
            # Safety pause tracking
            safety_halted = False
            pause_start_time = None
            pause_duration_total = 0.0

            while keep_running:   # 루프 시작
                # start control iteration
                # t_start = rtde_c.initPeriod()   # 바꾸기! 

                # send command to robot
                t_now = time.monotonic()
                # diff = t_now - pose_interp.times[-1]
                # if diff > 0:
                #     print('extrapolate', diff)
                # vel = 0.5
                # acc = 0.5
                # assert rtde_c.servoL(pose_command,   # 로봇 제어 부분! 바꾸기!
                #     vel, acc, # dummy, not used by ur5
                #     dt, 
                #     self.lookahead_time, 
                #     self.gain)

                
                # print("[DEBUG] curr_pose: ", curr_pose)
                # print("[DEBUG] pose_command: ", pose_command)


                # RB10 제어 - Get joint state from ROS topic subscription
                with self.joint_state_lock:
                    if self.joint_state is not None:
                        current_joint = self.joint_state.copy()
                    else:
                        # Fallback to API if no ROS message received
                        j = GetCurrentJoint()
                        current_joint = np.array([j.j0, j.j1, j.j2, j.j3, j.j4, j.j5]) * np.pi / 180   # rad
                
                curr_se3 = rb10.fkine(current_joint)   # m, rad (SE3)
                curr_pose = se3_to_pos_rotvec(curr_se3)   

                # Safety check: halt robot if filtered_distance is below threshold
                with self.filtered_distance_lock:
                    if self.filtered_distance is not None:
                        is_safe = np.all(self.filtered_distance >= self.safety_threshold)
                        
                        if not is_safe and not safety_halted:
                            # Just became unsafe - start pause
                            min_distance = np.min(self.filtered_distance)
                            print(f"[SAFETY] Filtered distance {min_distance:.4f} m below threshold {self.safety_threshold:.4f} m. Halting robot.")
                            MotionHalt()
                            safety_halted = True
                            pause_start_time = t_now
                        elif not is_safe and safety_halted:
                            # Still unsafe - accumulate pause time
                            if pause_start_time is not None:
                                pause_duration_total += (t_now - pause_start_time)
                                pause_start_time = t_now
                        elif is_safe and safety_halted:
                            # Just became safe - resume
                            if pause_start_time is not None:
                                pause_duration_total += (t_now - pause_start_time)
                                pause_start_time = None
                            print(f"[SAFETY] Filtered distance above threshold. Resuming robot movement.")
                            safety_halted = False
                
                # Skip robot commands if safety halted
                if not safety_halted:
                    # 매니퓰레이터 및 그리퍼 제어                
                    # Adjust t_now by pause duration to account for paused time
                    adjusted_t_now = t_now - pause_duration_total
                    pose_command = pose_interp(adjusted_t_now)   # 보간 해놓고 현재 시간의 목표 pose 가져옴
                    servoJ_rb(rb10, current_joint, pose_command[:6], dt)   # servoJ (3 pos + 3 rotvec)
                    # ServoL(pose_command)                                 # servoL
                    # Control gripper (pose_command[6] is gripper value after converting 6D rot to 3D rotvec)
                    if len(pose_command) >= 7:
                        self.ros2_node.gripper_control([pose_command[6]])


                # 현재 State 저장
                # update robot state; ringbuffer에 state 저장
                state = dict()
                # for key in self.receive_keys:
                #     state[key] = np.array(getattr(rtde_r, 'get'+key)())
                


                for key in self.receive_keys:
                    if key == 'robot_eef_pos':
                        state[key] = np.array(curr_pose[:3])   # 현재 pose; meter
                    elif key == 'robot_eef_quat':
                        # state[key] = np.array(smb.r2q(curr_pose[3:6]))   # 현재 quat
                        rot_vec = np.array(curr_pose[3:6])
                        quat = R.from_rotvec(rot_vec).as_quat()   # rot_vec --> quat
                        state[key] = np.array(quat)
                    elif key == 'robot_gripper_qpos':
                        # Get current gripper position from subscription
                        with self.gripper_pos_lock:
                            if self.gripper_pos is not None:
                                state[key] = np.array([float(self.gripper_pos)])
                            else:
                                state[key] = np.array([2100.0])  # Fallback to open state

                state['robot_receive_timestamp'] = time.time()
                self.ring_buffer.put(state)   


                # fetch command from queue
                try:
                    commands = self.input_queue.get_all()   # command 긁어옴
                    # print('[DEBUG] commands', commands)
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):   # 가져온 cmd 수만큼 실행
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.STOP.value:   # STOP: 그만두기
                        keep_running = False
                        # stop immediately, ignore later commands
                        break
                    # elif cmd == Command.SERVOL.value:   # SERVOL: target_pose 실행 
                    #     # since curr_pose always lag behind curr_target_pose
                    #     # if we start the next interpolation with curr_pose
                    #     # the command robot receive will have discontinouity 
                    #     # and cause jittery robot behavior.
                    #     target_pose = command['target_pose']  
                    #     duration = float(command['duration']) 
                    #     curr_time = t_now + dt
                    #     t_insert = curr_time + duration
                    #     pose_interp = pose_interp.drive_to_waypoint(
                    #         pose=target_pose,
                    #         time=t_insert,
                    #         curr_time=curr_time,
                    #         max_pos_speed=self.max_pos_speed,
                    #         max_rot_speed=self.max_rot_speed
                    #     )
                    #     last_waypoint_time = t_insert
                    #     if self.verbose:
                    #         print("[RTDEPositionalController] New pose target:{} duration:{}s".format(
                    #             target_pose, duration))
                            
                    # 이걸로 제어 (n_cmd 1개씩 제어)
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = command['target_pose']   # abs; 3d pose, 6d rotation, 1d gripper (10D)

                        # print('[DEBUG] target_pose_before', target_pose)

                        # target_pose[:3] = target_pose[:3] * 1000.0   # m -> mm
                        
                        target_position = target_pose[:3]   # 3d position, meter
                        target_rotvec = rot6d_to_rotvec(target_pose[3:9])   # 6d rotation -> 3d rot_vec
                        target_gripper = target_pose[9:10]  # gripper (keep as array for concatenation)
                        target_pose = np.concatenate([target_position, target_rotvec, target_gripper])   # 3d position, 3d rot_vec, 1d gripper (7D)

                        print('[DEBUG] target_pose', target_pose)
                        
                        # print('[DEBUG] current rot_vec', curr_pose[3:6])
                        # target_pose[:3] = curr_pose[:3] + target_pose[:3] * MAX_TRANS   # pose, meter
                        # target_pose[3:6] = (R.from_rotvec(target_pose[3:6] * MAX_ROT) * R.from_rotvec(curr_pose[3:6])).as_rotvec()   # rotation, rad
                        # target_pose[6] = curr_pose[6] + target_pose[6] * MAX_GRIP    # gripper

                        # print('[DEBUG] target rot_vec', target_pose[3:6])

                        target_time = float(command['target_time'])   # time.time 기준
                        # translate global time to monotonic time (absolute time)
                        target_time = time.monotonic() - time.time() + target_time   # time.monotonic 기준
                        # schedule_waypoint expects all times in absolute time (same frame as interpolator)
                        # Use virtual time for curr_time calculation, then convert back to absolute
                        curr_time_virtual = (t_now - pause_duration_total) + dt
                        curr_time_absolute = curr_time_virtual + pause_duration_total  # Convert virtual to absolute
                        pose_interp = pose_interp.schedule_waypoint(   # 여기서 pose_interp 갱신
                            pose=target_pose,
                            time=target_time,  # absolute time
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed,
                            curr_time=curr_time_absolute,  # absolute time
                            last_waypoint_time=last_waypoint_time  # absolute time
                        )
                        # Store last_waypoint_time in absolute time (from interpolator's internal times)
                        last_waypoint_time = pose_interp.times[-1]
                    else:
                        keep_running = False
                        break
                
                # regulate frequency
                # rtde_c.waitPeriod(t_start)
                # Note: ROS2 callbacks are processed automatically by executor in background thread
                t_elapsed = time.monotonic() - t_start
                sleep_time = dt - t_elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)   # 수동 제어 주기

                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()   # 준비완료; wait하던거 실행됨
                iter_idx += 1

                if self.verbose:
                    print(f"[RTDEPositionalController] Actual frequency {1/(time.perf_counter() - t_start)}")

        finally:
            # mandatory cleanup
            # decelerate
            # rtde_c.servoStop()   # 끝내기; 바꾸기!

            # # terminate
            # rtde_c.stopScript()
            # rtde_c.disconnect()
            # rtde_r.disconnect()
            MotionHalt()
            DisConnectToCB()
            
            # Cleanup ROS2
            if hasattr(self, 'executor'):
                self.executor.shutdown()
            if hasattr(self, 'ros2_node'):
                self.ros2_node.destroy_node()
            rclpy.shutdown()
            if hasattr(self, 'executor_thread'):
                self.executor_thread.join(timeout=1.0)

            self.ready_event.set()

            if self.verbose:
                print(f"[RTDEPositionalController] Disconnected from robot: {robot_ip}")
