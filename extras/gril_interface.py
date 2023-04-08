# Code developed by MD Nazmus Samin Sunbeam
import pygame
import pygame_gui
import numpy as np

import airsim
import cv2
import time
import os
import subprocess
import math
import pandas as pd
import timeit

import tensorflow as tf
from agil_airsim import arilNN
from losses import action_loss

import random
from xbox_controller_test import XboxController

from subscribe import CustomThread, gaze_run, gaze_call
from utils_eyetracking import Tobii4C_Cpp


import csv

header = ['rgb_addr', 'depth_addr', 'gaze_x', 'gaze_y', 'act_roll', 'act_pitch', 'act_yaw', 'act_throttle']
data = []

customObjects = {
    'action_loss': action_loss
}

def _toEulerianAngle(q):
        z = q.z_val
        y = q.y_val
        x = q.x_val
        w = q.w_val
        ysqr = y * y

        # roll (x-axis rotation)
        t0 = +2.0 * (w*x + y*z)
        t1 = +1.0 - 2.0*(x*x + ysqr)
        roll = math.atan2(t0, t1)

        # pitch (y-axis rotation)
        t2 = +2.0 * (w*y - z*x)
        if (t2 > 1.0):
            t2 = 1
        if (t2 < -1.0):
            t2 = -1.0
        pitch = math.asin(t2)

        # yaw (z-axis rotation)
        t3 = +2.0 * (w*z + x*y)
        t4 = +1.0 - 2.0 * (ysqr + z*z)
        yaw = math.atan2(t3, t4)

        return (pitch, roll, yaw)

# set random seed
random.seed(42)

# initialize joystick object
joy = XboxController()
eyetracker = Tobii4C_Cpp()

aril_model = "gril.h5"

aril = tf.keras.models.load_model(aril_model, custom_objects=customObjects)
##########################################################################

pygame.init()
pygame.display.set_caption('CoL Experiment Interface')
display = pygame.display.set_mode((800, 600))
# x = np.arange(0, 224)
# y = np.arange(0, 224)
# X, Y = np.meshgrid(x, y)
# Z = X + Y
# Z = 255*Z/Z.max()
manager = pygame_gui.UIManager((800, 600))

# connect to AirSim client
client = airsim.MultirotorClient()
client.confirmConnection()



# UI elements ################################################

# text entries 
x_a = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((10, 250), (50, 30)),
                                             manager=manager)
y_a = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((70, 250), (50, 30)),
                                             manager=manager)
z_a = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((130, 250), (50, 30)),
                                             manager=manager)
q_z = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((10, 290), (50, 30)),
                                             manager=manager)
# buttons
rollout_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((250, 10), (140, 50)),
                                             text='Start Rollout',
                                             manager=manager)
end_rollout_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((250, 70), (140, 50)),
                                             text='End Rollout',
                                             manager=manager)
datacollection_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((250, 130), (200, 50)),
                                             text='Start Data Collection',
                                             manager=manager)
end_datacollection_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((250, 190), (200, 50)),
                                             text='End Data Collection',
                                             manager=manager)
reset_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((650, 10), (140, 50)),
                                             text='Reset',
                                             manager=manager)
# labels
agent_position_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((200, 250), (200, 50)),
                                             text='Agent Position',
                                             manager=manager)


#################################################################
clock = pygame.time.Clock()
running = True
gril_control = False

# get the image of the scene
sc = 10
duration = 1e-0
done = False
img_counter = 0

client.enableApiControl(True)

client.armDisarm(True)
client.takeoffAsync().join()

# just hover
client.hoverAsync().join()

# teleportation/spawning functions for QUADROTOR
start_location = 7
inter_no = 553545
rgb_addr = 'stat_truck_l{}/fly{}/rgb'.format(start_location, inter_no)
depth_addr = 'stat_truck_l{}/fly{}/depth'.format(start_location, inter_no)
os.makedirs(rgb_addr)
os.makedirs(depth_addr)
pose = client.simGetVehiclePose()
pose.position.x_val += 30
pose.position.y_val += 20
# pose.position.z_val = 10
pose.orientation.z_val -= 1

client.simSetVehiclePose(pose, True, "SimpleFlight")


# gaze = subprocess.run(["dzdo",  "./main_2"],  # <-- Change to whatever the compiled cpp file is called
#         stdout=subprocess.PIPE,
#         stderr=subprocess.PIPE)

# thread = CustomThread(target=gaze_run(gaze))
# thread.start()
 
while running:
   
    

    time_delta = clock.tick(60)/1000.0

    kairos = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
    kairos_bstr = kairos[0]

    # get numpy array
    img1d = np.fromstring(kairos_bstr.image_data_uint8, dtype=np.uint8)

    # reshape to 4-channel image (for Unreal 4.18)
    #img_rgb = img1d.reshape(kairos_bstr.height, kairos_bstr.width, 4)
        
    # reshape to 3-channel image (for Unreal 4.25)
    img_viz = img1d.reshape(kairos_bstr.height, kairos_bstr.width, 3)
    img_viz = cv2.rotate(img_viz, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img_viz = cv2.cvtColor(img_viz, cv2.COLOR_BGR2RGB)
    img_viz = cv2.flip(img_viz, 0)
    # print(np.min(img_rgb))
    Z = img_viz
    surf = pygame.surfarray.make_surface(Z)
    pose = client.simGetVehiclePose()

     # read joystick inputs
    joy_cmd = joy.read()
        
    # # reshape to 3-channel image (for Unreal 4.25)
    # img_rgb = img1d.reshape(kairos_bstr.height, kairos_bstr.width, 3)
    # img_rgb = cv2.rotate(img_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    # img_rgb = cv2.flip(img_rgb, 0)
    # print(np.min(img_rgb))
    pose = client.simGetVehiclePose()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame_gui.UI_TEXT_ENTRY_FINISHED:
            if event.ui_element == x_a:
                # print('Changed text:', event.text)
                pose.position.x_val += float(event.text)
                client.simSetVehiclePose(pose, True, "SimpleFlight")
            if event.ui_element == y_a:
                pose.position.y_val += float(event.text)
                client.simSetVehiclePose(pose, True, "SimpleFlight")
            if event.ui_element == z_a:
                pose.position.z_val += float(event.text)
                client.simSetVehiclePose(pose, True, "SimpleFlight")
            if event.ui_element == q_z:
                pose.orientation.z_val += float(event.text)
                client.simSetVehiclePose(pose, True, "SimpleFlight")
	 

        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == rollout_button:
                gril_control = True

            if event.ui_element == end_rollout_button:
                gril_control = False
                
            if event.ui_element == datacollection_button:
                print('Start!')
            if event.ui_element == end_datacollection_button:
                print('End!')
            if event.ui_element == reset_button:
                client.reset()
    

        manager.process_events(event)
    
    if gril_control:
        kairos = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
        kairos_bstr = kairos[0]

        # get numpy array
        img1d = np.fromstring(kairos_bstr.image_data_uint8, dtype=np.uint8)

        # reshape to 4-channel image (for Unreal 4.18)
        #img_rgb = img1d.reshape(kairos_bstr.height, kairos_bstr.width, 4)
            
        # reshape to 3-channel image (for Unreal 4.25)
        img_viz = img1d.reshape(kairos_bstr.height, kairos_bstr.width, 3)
        img_viz = cv2.rotate(img_viz, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img_viz = cv2.cvtColor(img_viz, cv2.COLOR_BGR2RGB)
        img_viz = cv2.flip(img_viz, 0)
        # print(np.min(img_rgb))
        
        pose = client.simGetVehiclePose()


        # client.enableApiControl(True)

        # client.armDisarm(True)
        # client.takeoffAsync().join()

        # # just hover
        # client.hoverAsync().join()

        
        # getting quad states
        state = client.getMultirotorState()

        # convert from quaternion to euler angles
        (pitch, roll, yaw) = _toEulerianAngle(state.kinematics_estimated.orientation)
        # getting images
        kairos = client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Scene, False, False),
                                airsim.ImageRequest(0, airsim.ImageType.DepthVis, True)])
        kairos_bstr = kairos[0]
        kairos_depth = kairos[1]

        # get numpy array
        img1d = np.fromstring(kairos_bstr.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(kairos_bstr.height, kairos_bstr.width, 3)

        img_viz = img1d.reshape(kairos_bstr.height, kairos_bstr.width, 3)
        img_viz = cv2.rotate(img_viz, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img_viz = cv2.cvtColor(img_viz, cv2.COLOR_BGR2RGB)
        img_viz = cv2.flip(img_viz, 0)

        
        # convert from bgr to rgb
        # img_rgb = np.fliplr(img_rgb.reshape(-1,3)).reshape(img_rgb.shape)

        
        # reshape to 3-channel image (for Unreal 4.25)
        dp1d = np.array(kairos_depth.image_data_float, dtype=np.float32)
        dp = dp1d.reshape(kairos_depth.height, kairos_depth.width)
        img_depth = np.array(np.abs(1-dp) * 255, dtype=np.uint8)
        
        
        output, gaze = arilNN(img_rgb, img_depth, aril)
        # print(224*gaze[:,0], 224*gaze[:,1])
        # gaze_coord = np.array((224 * gaze[:,0]), (224 * gaze[:,1]))

        # img_viz = cv2.circle(img_viz, (int(224*gaze[:,0]), int(224*gaze[:,1])), 5, (255, 0, 0), 2)   
        


        if joy_cmd[4]:
            # client.enableApiControl(False)
            act_roll = float(joy_cmd[2])
            act_pitch = float(joy_cmd[3])
            act_throttle = float(joy_cmd[1])
            act_yaw = float(joy_cmd[0])
            
            gaze_x, gaze_y = eyetracker.read_gaze()
            # gaze_x = eyetracker.gaze_x
            # gaze_y = eyetracker.gaze_y
            print(gaze_x, gaze_y)
            x = gaze_x
            y = gaze_y

            # thread = CustomThread(target=gaze_run)
            # y, x = gaze_run(gaze)
            # y, x = thread.join()
            # thread.start()
            # y, x = gaze_call()
            # time.sleep(.5)
            img_viz = cv2.circle(img_viz, (int(224*x), int(224*y)), 5, (0, 0, 255), 2)
            # rgb_addr   = 'moving_truck/fly/rgb/rgb_{}.png'.format(img_counter)
            # depth_addr = 'moving_truck/fly/depth/depth_{}.png'.format(img_counter)
            
            cv2.imwrite(os.path.join(rgb_addr, 'rgb_{}.png'.format(img_counter)), img_rgb)
            cv2.imwrite(os.path.join(depth_addr, 'depth_{}.png'.format(img_counter)), img_depth)
            
            row = [rgb_addr, depth_addr, x, y, act_roll, act_pitch, act_yaw, act_throttle]
            # row = [x, y, act_roll, act_pitch, act_yaw, act_throttle]
            data.append(row)
            img_counter += 1
            with open('stat_truck_l{}/fly{}/log.csv'.format(start_location, inter_no), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                # write the header
                writer.writerows(data)
            

        else:
            # client.enableApiControl(True)
            # output = agilNN(gaze, agil, img)
            act_roll     = float(output[:,0]) 
            act_pitch    = float(output[:,1]) 
            #throttle = abs(float(output[:,2]))
            # act_throttle = (float(output[:,2]) + 1.0)/2.0 + .15
            act_throttle = float(output[:,2])
            act_yaw      = float(output[:,3])
            img_viz = cv2.circle(img_viz, (int(224*gaze[:,0]), int(224*gaze[:,1])), 5, (255, 0, 0), 2) 

   

        vx = sc / 1.5 * act_pitch #1
        vy = sc / 1.5 * act_roll #0
        vz = 10 * sc * act_yaw #3
        ref_alt = state.kinematics_estimated.position.z_val + sc / 2 * act_throttle#2

        # translate from inertial to body frame
        C = np.zeros((2, 2))
        C[0, 0] = np.cos(yaw)
        C[0, 1] = -np.sin(yaw)
        C[1, 0] = -C[0, 1]
        C[1, 1] = C[0, 0]
        vb = C.dot(np.array([vx, vy]))

        # send commands
        client.moveByVelocityZAsync(
            vb[0],
            vb[1],
            ref_alt,
            duration,
            airsim.DrivetrainType.MaxDegreeOfFreedom,
            airsim.YawMode(True, vz),
        )

        Z = img_viz
        surf = pygame.surfarray.make_surface(Z)
    
    
    manager.update(time_delta)
    
    display.blit(surf, (10, 10))
    manager.draw_ui(display)
    pygame.display.update()

bash_folder = "dzdo chmod -R 777 /stat_truck_l{}".format(start_location)
subprocess.run(bash_folder.split(), shell=True) 