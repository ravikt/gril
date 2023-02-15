import airsim
import cv2
import numpy as np
import time
import tensorflow as tf
import os
from agil_airsim import arilNN
import math
import timeit
import argparse

from losses import action_loss
from airsim_utils import AirSimEnv

customObjects = {
    'action_loss': action_loss
}

aril_model = "gril.h5"
aril = tf.keras.models.load_model(aril_model, custom_objects=customObjects)

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

for i in range(EPISODES):

    done = False
    while not done:

        img_rgb, img_depth =  env.getRGBImage(), env.getRGBDepthImage()
        output = arilNN(img_rgb, img_depth, aril)
        roll     = float(output[:,0]) 
        pitch    = float(output[:,1]) 
        throttle = float(output[:,2])
        yaw      = float(output[:,3])      
        vx, vy, vz, ref_alt = env.angularRatesToLinearVelocity(pitch, roll, yaw, throttle, SC)
        vb = env.inertialToBodyFrame(yaw, vx, vy)
        env.controlQuadrotor(vb, vz, ref_alt, DURATION)
