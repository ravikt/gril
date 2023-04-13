import airsim
import numpy as np
import cv2
import argparse
import sys
import tensorflow as tf

from agil_airsim import grilNN
from airsim_utils import AirSimEnv
from losses import action_loss

# environment initialization
env = AirSimEnv()
env.connectQuadrotor()
env.enableAPI(True)
env.armQuadrotor()
env.takeOff()
env.hover()

# load model
customObjects = {
    'action_loss': action_loss
}

saved_model = "gril.h5"
model = tf.keras.models.load_model(saved_model, custom_objects=customObjects)

# useful terminal flags
parser = argparse.ArgumentParser(
                    prog = 'Experiment',
                    description = 'Configurations for the experiment',
                    epilog = 'Configuration include the number of episodes')
parser.add_argument('-e', '--episodes', type=int, help='Number of episodes to run', default=10)
parser.add_argument('-d', '--duration', type=int, help='Duration of control command', default=1)
parser.add_argument('-sc', '--sc', type=int, help='Constant related to ang->lin', default=10)
args = parser.parse_args()

print(args)

# experiment parameters
EPISODES=args.episodes
DURATION=args.duration
SC=args.sc

for i in range(EPISODES):
    done = False
    env.raviktTeleportQuadrotor(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]) # x, y, z, and yaw [-1 to 1]
    while not done:
        
        img_rgb, img_depth = env.getRGBImage(), env.getDepthImage()
        # GRIL agent
        output = grilNN(img_rgb, img_depth, model)
        roll     = float(output[:,0])
        pitch    = float(output[:,1])
        throttle = float(output[:,2])
        yaw      = float(output[:,3])
        vx, vy, vz, ref_alt = env.angularRatesToLinearVelocity(pitch, roll, yaw, throttle, SC)
        vb = env.inertialToBodyFrame(yaw, vx, vy)
        env.controlQuadrotor(vb, vz, ref_alt, DURATION)




    
