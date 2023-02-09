import airsim
import numpy as np
import cv2

from airsim_utils import AirSimEnv

env = AirSimEnv()
env.connectQuadrotor()
env.enableAPI(True)

EPISODES=10

for i in range(10):

    done = False
    while not done:

    