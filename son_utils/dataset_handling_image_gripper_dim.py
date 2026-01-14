import h5py
import numpy as np
import os

file_path = '/home/ws/Desktop/robotory_rb10_ros2/scripts/idp_data_by_kwon.hdf5'

if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
else:
    with h5py.File(file_path, 'r+') as f:
        if 'data' in f:
            data_group = f['data']
            demo_keys = list(data_group.keys())
            print(f"Checking {len(demo_keys)} demo groups...")
            
            reshape_count = 0
            rename_count = 0
            
            for demo_key in demo_keys:
                # Access demo group (e.g., data/demo_0)
                demo_node = data_group[demo_key]
                
                # Logic: Find and rename observation group to 'obs'
                if 'obs' not in demo_node and 'observations' in demo_node:
                    # Rename 'observations' to 'obs' for consistency
                    demo_node.move('observations', 'obs')
                    rename_count += 1
                
                # Process if 'obs' group exists (either originally or renamed)
                if 'obs' in demo_node:
                    obs_group = demo_node['obs']
                    
                    if 'gripper' in obs_group:
                        gripper_data = obs_group['gripper'][:]
                        
                        # Reshape if dimension is (N,) -> (N, 1)
                        if gripper_data.ndim == 1:
                            new_data = gripper_data.reshape(-1, 1)
                            
                            # Safe replacement process
                            if 'gripper_tmp' in obs_group:
                                del obs_group['gripper_tmp']
                            
                            obs_group.create_dataset('gripper_tmp', data=new_data)
                            del obs_group['gripper']
                            obs_group.move('gripper_tmp', 'gripper')
                            
                            reshape_count += 1
                        elif gripper_data.ndim == 2 and gripper_data.shape[1] == 1:
                            pass
                        else:
                            print(f"Warning [{demo_key}]: Unexpected shape {gripper_data.shape}")

            print("-" * 30)
            print(f"Success: Renamed {rename_count} observation groups.")
            print(f"Success: Reshaped 'gripper' in {reshape_count} demos.")
            
            # Verification of the first demo result
            if len(demo_keys) > 0:
                test_key = demo_keys[0]
                if 'obs' in data_group[test_key]:
                    final_shape = data_group[test_key]['obs']['gripper'].shape
                    print(f"Verification [{test_key}]: gripper shape = {final_shape}")
        else:
            print("Error: 'data' group not found.")