import h5py

file_path = '/home/ws/Downloads/RB10_dataset/pick_and_place/idp_data.hdf5'

with h5py.File(file_path, 'r+') as f:
    # 1. 'data' 그룹 접근
    if 'data' in f:
        data_group = f['data']
        print(f"총 {len(data_group)}개의 demo 그룹을 찾았습니다. 변경을 시작합니다...")
        
        changed_count = 0
        
        # 2. 'data' 그룹 내의 모든 키(demo_0, demo_1...)를 순회
        for demo_key in data_group.keys():
            # 각 데모 그룹 객체 (예: f['data/demo_0'])
            demo_group = data_group[demo_key]
            
            # 3. 해당 데모 그룹 안에 'actions'가 있는지 확인
            if 'action' in demo_group:
                # 덮어쓰기 방지: 이미 'action'으로 바뀐 적이 없는지 체크
                if 'actions' not in demo_group:
                    demo_group.move('action', 'actions')
                    changed_count += 1
                else:
                    print(f"Skip [{demo_key}]: 이미 'action' 키가 존재합니다.")
        
        print(f"\n완료: 총 {changed_count}개의 그룹에서 'actions'를 'action'으로 변경했습니다.")
        
        # 확인용: 첫 번째 데모의 키 출력
        first_demo = list(data_group.keys())[0]
        print(f"확인 ({first_demo}): {list(data_group[first_demo].keys())}")
        
    else:
        print("오류: 파일 내 최상위 'data' 그룹을 찾을 수 없습니다.")