import airsim
import numpy as np
import cv2
import argparse

from airsim_utils import AirSimEnv

env = AirSimEnv()

env.connectQuadrotor()
env.enableAPI(True)
env.armQuadrotor()
env.takeOff()
env.hover()

parser = argparse.ArgumentParser(
                    prog = 'Experiment',
                    description = 'Configurations for the experiment',
                    epilog = 'Configuration include the number of episodes')
parser.add_argument('-e', '--episodes', type=int, help='Number of episodes to run', default=10)
parser.add_argument('-d', '--duration', type=int, help='Duration of control command', default=1)
parser.add_argument('-sc', '--sc', type=int, help='Constant related to ang->lin', default=10)

# Experiment parameters
EPISODES=parser.episodes
DURATION=parser.duration
SC=parser.sc

for i in range(EPISODES):

    done = False
    while not done:

        env.getQuadrotorState()
        env.getRGBImage()
        pitch, roll, yaw, throttle = (np.random.rand()*10, np.random.rand()*10, np.random.rand()*10, np.random.rand()*10)
        vx, vy, vz, ref_alt = env.angularRatesToLinearVelocity(pitch, roll, yaw, throttle, SC)
        vb = env.inertialToBodyFrame(yaw, vx, vy)
        env.controlQuadrotor(vb, vz, ref_alt, DURATION)


    