import h5py

file_path = '/home/ws/Desktop/robotory_rb10_ros2/scripts/idp_data_by_kwon.hdf5'

with h5py.File(file_path, 'r+') as f:
    if 'data' in f:
        data_group = f['data']
        print(f"총 {len(data_group)}개의 demo 그룹을 처리합니다...")
        
        count_wrist = 0
        count_scene = 0
        
        for demo_key in data_group.keys():
            # 1. 'obs' 그룹으로 접근 (예: data/demo_0/obs)
            # obs가 그룹인지 확인하고 접근합니다.
            if 'obs' in data_group[demo_key]:
                obs_group = data_group[demo_key]['obs']
                
                # 2. image_wrist -> image0 변경
                if 'image_wrist' in obs_group:
                    if 'image0' not in obs_group:
                        obs_group.move('image_wrist', 'image0')
                        count_wrist += 1
                
                # 3. image_scene -> image1 변경
                if 'image_scene' in obs_group:
                    if 'image1' not in obs_group:
                        obs_group.move('image_scene', 'image1')
                        count_scene += 1
        
        print("-" * 30)
        print(f"완료: 'image_wrist' -> 'image0' 변경: {count_wrist}건")
        print(f"완료: 'image_scene' -> 'image1' 변경: {count_scene}건")
        
        # 결과 확인 (첫 번째 데모의 obs 내부 키 출력)
        first_demo = list(data_group.keys())[0]
        first_obs = data_group[first_demo]['obs']
        print(f"\n확인 [{first_demo}/obs]: {list(first_obs.keys())}")

    else:
        print("오류: 'data' 그룹을 찾을 수 없습니다.")