import airsim
import numpy as np
import cv2
import argparse
import sys

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
args = parser.parse_args()

print(args)

# Experiment parameters
EPISODES=args.episodes
DURATION=args.duration
SC=args.sc
sys.argv[1], sys.argv[2]
for i in range(EPISODES):
    done = False
    env.raviktTeleportQuadrotor(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]) # x, y, z, and yaw [-1 to 1]
    while not done:
        
        env.getRGBImage()
        # random agent
        pitch, roll, yaw, throttle = (np.random.rand()*10, np.random.rand()*10, np.random.rand()*10, np.random.rand()*10)
        vx, vy, vz, ref_alt = env.angularRatesToLinearVelocity(pitch, roll, yaw, throttle, SC)
        vb = env.inertialToBodyFrame(yaw, vx, vy)
        env.controlQuadrotor(vb, vz, ref_alt, DURATION)




    
