import numpy as np

# 원본 -1~1 정규화된 델타 회전벡터
rot_delta_norm = np.array([0.96183622, 0.43046194, -0.53439486])

# 1) 스케일 복원: π 곱하기
rot_delta = rot_delta_norm * np.pi

# 2) 회전각 (rad) = 벡터 노름
angle_rad = np.linalg.norm(rot_delta)

# 3) 회전각 (deg)
angle_deg = np.degrees(angle_rad)

# 4) 회전축 (unit vector)
axis = rot_delta / angle_rad if angle_rad != 0 else np.array([1.,0.,0.])

print(f"회전각: {angle_rad:.3f} rad  =  {angle_deg:.1f}°")
print(f"회전축: [{axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}]")