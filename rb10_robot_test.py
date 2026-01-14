import rospy
import spatialmath.base as smb
from std_msgs.msg import Int32, Float64, String
from diffusion_policy_test.msg import OnRobotRGOutput, OnRobotRGInput
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


ToCB(ip="192.168.111.50")
rb10 = RB10()
CobotInit()

# Real or Simulation
# SetProgramMode(PG_MODE.REAL)
SetProgramMode(PG_MODE.SIMULATION)



j = GetCurrentJoint()
current_joint = np.array([j.j0, j.j1, j.j2, j.j3, j.j4, j.j5]) * np.pi / 180   # rad
curr_se3 = rb10.fkine(current_joint)   # m, rad (SE3)
print("Current Joint:", current_joint)
p = GetCurrentTCP()