import numpy as np

rot6d = np.array([np.cos(np.pi/3), np.sin(np.pi/3), 0, -np.sin(np.pi/3), np.cos(np.pi/3), 0])
a1 = rot6d[:3]
a2 = rot6d[3:]

b1 = a1 / np.linalg.norm(a1)

a2_proj = np.dot(b1, a2) * b1
b2 = a2 - a2_proj
b2 = b2 / np.linalg.norm(b2)

b3 = np.cross(b1, b2)

R_mat = np.stack((b1, b2, b3), axis=1)  

print(R_mat)