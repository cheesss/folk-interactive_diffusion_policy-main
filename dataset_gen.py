import os
import sys
import h5py
import argparse
import time
import cv2
import threading

import numpy as np
import pyrealsense2 as rs

from pynput import keyboard
from datetime import datetime
from api.cobot import *

from scipy.spatial.transform import Rotation as R
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import JointState
from rclpy.logging import get_logger

from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import UInt8, Int32
from rclpy.duration import Duration

today = datetime.now().strftime('%m%d_%H%M')
# data_dir = f'/home/vision/dualarm_ws/src/doosan-robot2/dsr_example2/dsr_dataset/{today}'
root_dir = os.path.expanduser('~')
data_dir = f'{root_dir}/rb_ws/src/robotory_rb10_ros2/data/{today}'
if not os.path.isdir(data_dir):
    os.makedirs(data_dir)

# HDF5 file path
hdf5_path = os.path.join(data_dir, 'common_data.hdf5')

_logger = get_logger('dataset_gen')

def init_buffer():   
    return {
        'observations': {
            'joint': [], # abs, rad (6)
            'image_wrist': [], # D405  (640x480) (BGR) (wrist)
            'image_scene': [], # D435  (640x480) (BGR) (scene)
            'gripper': [] # gripper position (1)
        }
    }   # 20 Hz

def get_device_serials():
    ctx = rs.context()
    serials = []
    for device in ctx.query_devices():
        serials.append(device.get_info(rs.camera_info.serial_number))
    return serials


def on_press(key):
    global recording, terminal, dataset_generator, homing, prompt_save
    try:
        if key.char == 's':
            if not recording:
                recording = True
                print("Start recording")
                if homing:
                    dataset_generator.send_teleop_command(1)
            
        elif key.char == 'q':
            if recording:
                recording = False
                prompt_save = True  # Flag to trigger save prompt
                print("Stop recording")
                if homing:
                    dataset_generator.send_teleop_command(2)
                
        elif key.char == 't':
            terminal = True
    except AttributeError:
        pass

def make_demo_n(buffer, hdf5_path):
    """
    Save a single demo to HDF5 file.
    Opens file in append mode, adds the demo, and closes immediately.
    """
    # Open file in append mode (or create if doesn't exist)
    if os.path.exists(hdf5_path):
        with h5py.File(hdf5_path, 'a') as f:
            if 'data' not in f:
                data = f.create_group('data')
            else:
                data = f['data']
            
            # Find next demo index
            n = len(data.keys())
            demo_n = data.create_group(f'demo_{n}')
            obs = demo_n.create_group('observations')

            for name, values in buffer['observations'].items():
                obs.create_dataset(name, data=np.array(values))

            print(f"Saved demo_{n} with {len(buffer['observations']['joint'])} steps to {hdf5_path}")
    else:
        # Create new file
        with h5py.File(hdf5_path, 'w') as f:
            data = f.create_group('data')
            demo_n = data.create_group('demo_0')
            obs = demo_n.create_group('observations')

            for name, values in buffer['observations'].items():
                obs.create_dataset(name, data=np.array(values))

            print(f"Created new file and saved demo_0 with {len(buffer['observations']['joint'])} steps to {hdf5_path}")

class Pipeline:
    def __init__(self, serial):
        self.serial = serial
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(serial)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(self.config)

        for i in range(3):
            try: 
                self.pipeline.wait_for_frames(timeout_ms=1000)
                # print(f"Realsense camera {serial} initialized. try {i}")
            except:
                self.pipeline.stop()
                self.pipeline.start(self.config)
                self.pipeline.wait_for_frames()
                # print(f"Realsense camera {serial} re-initialized. except {i}")

        _logger.info(f"Realsense camera {serial} initialized.")

    def get_frame(self):
        try:
            frame = self.pipeline.wait_for_frames(timeout_ms=50)
        except RuntimeError as e:
            _logger.warning(f"Realsense timeout: {e}")
            return None
        
        color_frame = frame.get_color_frame()
        if not color_frame:
            _logger.warning("No color frame from the camera")
            return None
        
        color_image = np.asanyarray(color_frame.get_data())

        return color_image

class DatasetGenerator(Node):
    def __init__(self):
        super().__init__('dataset_generator')
        
        # Create callback groups for parallel execution
        self.reentrant_group = ReentrantCallbackGroup()
        
        # Initialize state variables
        self.joint_state = None
        self.gripper_pos = None
        
        self.teleop_control_pub = self.create_publisher(
            UInt8,
            '/teleop_control',
            10
        )

        self.joint_state_subscriber = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10,
            callback_group=self.reentrant_group  # Allow parallel execution
        )

        self.gripper_pos_subscriber = self.create_subscription(
            Int32,
            '/gripper/present_position',
            self.gripper_pos_callback,
            10,
            callback_group=self.reentrant_group  # Allow parallel execution
        )

    def joint_state_callback(self, msg):
        self.joint_state = msg.position
        # self.get_logger().info(f"Joint state: {self.joint_state}")

    def gripper_pos_callback(self, msg):
        self.gripper_pos = msg.data
        # self.get_logger().info(f"Gripper position: {self.gripper_pos}")

    def send_teleop_command(self, command):
        msg = UInt8()
        msg.data = command
        self.teleop_control_pub.publish(msg)
        
        command_names = {0: "PAUSED", 1: "RUNNING", 2: "GO_HOME"}
        command_name = command_names.get(command, f"UNKNOWN({command})")
        print(f"Teleop command sent: {command_name} ({command})")

def show_images(images, window_names, recording=False):
    for name, img in zip(window_names, images):
        if img is None:
            continue
        overlay = img.copy()
        cv2.putText(
            overlay,
            'REC' if recording else 'LIVE',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255) if recording else (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        cv2.imshow(name, overlay)
    cv2.waitKey(1)

def main(args=None):
    global terminal, recording, latest_joint, latest_gripper, dataset_generator, homing
    recording = False
    terminal = False
    latest_joint = None
    latest_gripper = None

    connected_serials = get_device_serials()
    print("Connected serials: ", connected_serials)

    # 카메라 시리얼 넘버 설정 
    serials = [ '126122270795', '117322071192']   # Wrist, Scene
    # serials = ['117322071192']   # D435 (RB10)
    # serials = ['242422304502']   # D455
    # serials = ['242422304502', '126122270712']   # D455, D405
    # serials = None

    if serials == None:
        serials = get_device_serials()
    print("Selected serials: ", serials)

    assert all(serial in connected_serials for serial in serials), "Selected serials not connected"
    pipelines = [Pipeline(serial) for serial in serials]

    window_names = [f'wrist ({serials[0]})', f'scene ({serials[1]})']
    for w in window_names:
        cv2.namedWindow(w, cv2.WINDOW_NORMAL)

    assert len(pipelines) > 0, "No cameras found"

    # ROS2 Initialization
    rclpy.init(args=args)
    dataset_generator = DatasetGenerator()
    
    # Create MultiThreadedExecutor with 4 threads
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(dataset_generator)

    # Run executor in a separate thread to process callbacks at full rate
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    # Keyboard Listener
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # Save Period
    Hz = 20
    rate = 1.0 / Hz

    buffer = init_buffer()

    print('s: start, q: stop, t: terminate')
    print("Always terminate with 't', and change the save file name")
    print("Starting demo session...")

    if homing:
        print("Moving robots to preset position...")
        dataset_generator.send_teleop_command(2)
        
        # Wait for 5 seconds
        print("Waiting 5 seconds for robots to reach preset position...")
        time.sleep(5.0)

        print("Ready for demonstration! Press 's' to start recording.")
    else:
        print("Manual mode - homing should be operated manually.")
        print("Ready for demonstration! Press 's' to start recording.")

    
    try: 

        while rclpy.ok():
            start_time = time.monotonic()

            if terminal:
                print("Terminating.")
                # Check if there's unsaved data
                if len(buffer['observations']['joint']) > 0:
                    print("Warning: You have unsaved data!")
                    save_last = input("Save current buffer before exiting? (y/n): ").strip().lower()
                    if save_last == 'y':
                        try:
                            make_demo_n(buffer, hdf5_path)
                        except Exception as e:
                            print(f"Error saving demo: {e}")
                break

            images = [pipeline.get_frame() for pipeline in pipelines]
            show_images(images, window_names, recording=recording)

            if recording and dataset_generator.gripper_pos is not None and dataset_generator.joint_state is not None:

                joint = np.array(dataset_generator.joint_state) # rad
                gripper = dataset_generator.gripper_pos

                images = []
                for pipeline in pipelines:
                    images.append(pipeline.get_frame())

                buffer['observations']['joint'].append(joint)
                buffer['observations']['image_wrist'].append(images[0].copy())
                buffer['observations']['image_scene'].append(images[1].copy())
                buffer['observations']['gripper'].append(gripper)

            elif not recording and len(buffer['observations']['joint']) > 0:
                while True:
                    data_store = input("Store demo data? (y/n): ").strip().lower()
                    if data_store == 'y':
                        try:
                            make_demo_n(buffer, hdf5_path)
                        except Exception as e:
                            print(f"Error saving demo: {e}")
                            print("Demo not saved, but you can try again.")
                            continue
                        break
                    elif data_store == 'n':
                        print("Data discarded.")
                        break
                    else:
                        print("Invalid input.")

                buffer = init_buffer()


            else:
                images = []
                for pipeline in pipelines:
                    images.append(pipeline.get_frame())
                
            current_time = time.monotonic()
            elapsed_time = current_time - start_time
            reward_time = rate - elapsed_time
            if reward_time > 0:
                time.sleep(reward_time)
            
    except KeyboardInterrupt:
        pass
    
    finally:
        listener.stop()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        rclpy.shutdown()
        executor_thread.join(timeout=2.0)  # Wait for executor thread to finish
        for pipeline in pipelines:
            pipeline.pipeline.stop()
            print(f'pipeline stopped (serial number: {pipeline.serial})')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dual-arm data generation with optional auto homing control')
    parser.add_argument('-m', '--homing', action='store_true', 
                       help='Enable automatic homing mode switching (default: disabled)')
    args = parser.parse_args()
    
    homing = args.homing

    main()