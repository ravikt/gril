import airsim
import cv2
import numpy as np
import time
import tensorflow as tf
import os
from agil_airsim import arilNN
import math
import timeit
from losses import action_loss

# connect to AirSim client
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.simEnableWeather(True)

#client.simSetWeatherParameter(airsim.WeatherParameter.Rain, 1.0);
#client.simSetWeatherParameter(airsim.WeatherParameter.Snow, 1.0);
#client.simSetWeatherParameter(airsim.WeatherParameter.MapleLeaf, 1.0);
#client.simSetWeatherParameter(airsim.WeatherParameter.Dust, 0.25);
client.simSetWeatherParameter(airsim.WeatherParameter.Fog, 0.25)


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



aril_model = "gril.h5"

aril = tf.keras.models.load_model(aril_model, custom_objects=customObjects)

# rollout loop
img_counter = 0
#airsim.wait_key('Press any key to begin rollouts')
while(True):

    # arm and takeoff
    print("Taking off...")
    client.armDisarm(True)
    client.takeoffAsync().join()

    # just hover
    client.hoverAsync().join()

    # get the image of the scene
    sc = 10
    duration = 1e-0
    done = False
    
    # teleportation/spawning functions for QUADROTOR
    pose = client.simGetVehiclePose()
    #pose.position.x_val -= 25
    #pose.position.y_val += 25
    #pose.position.z_val = -4

    client.simSetVehiclePose(pose, True, "SimpleFlight")

    # # teleportation/spawning functions for YELLOW TRUCK
    # target_name = 'Truck_4'
    # target_pose = client.simGetObjectPose(target_name)
    # target_pose.position.x_val -= 20 
    # target_pose.position.y_val -= 10
    # # target_pose.position.z_val
    # client.simSetObjectPose(target_name, target_pose, teleport = True)
  
    while(not done):
        
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
        # convert from bgr to rgb
        # img_rgb = np.fliplr(img_rgb.reshape(-1,3)).reshape(img_rgb.shape)

        
        # reshape to 3-channel image (for Unreal 4.25)
        dp1d = np.array(kairos_depth.image_data_float, dtype=np.float32)
        dp = dp1d.reshape(kairos_depth.height, kairos_depth.width)
        img_depth = np.array(np.abs(1-dp) * 255, dtype=np.uint8)
        
        
        
        # for Unreal 4.18
        # The binaries from Vinicius' repo
        # img = img_rgb[:,:,0:3]
        
        # for Unreal 4.25
        
        # Print resized color image of the current frame
        #cv2.imwrite('out{}.png'.format(img_counter), cv2.resize(img_rgb, (224, 224)))
        #cv2.imwrite('dp{}.png'.format(img_counter), cv2.resize(img_depth, (224, 224)))
        
        # AGIL network predictions
        # roll, pitch, throttle, yaw
        # output = arilNN(img_rgb, dp, aril)
        output = arilNN(img_rgb, img_depth, aril)
        # output = agilNN(gaze, agil, img)
        act_roll     = float(output[:,0]) 
        act_pitch    = float(output[:,1]) 
        #throttle = abs(float(output[:,2]))

        # act_throttle = (float(output[:,2]) + 1.0)/2.0 + .15
        act_throttle = float(output[:,2])
        act_yaw      = float(output[:,3])           
        # print(roll, pitch, throttle, yaw)

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

        print(vb[0], vb[1], ref_alt, duration)
        #print(timeit.timeit('output = agilNN(gaze, agil, img))')
        # # actions mapped to controlling the quadrotor; args := (roll, pitch, yaw, throttle, duration)
        # client.moveByRollPitchYawThrottleAsync(roll, pitch, yaw, throttle, 1)

        # send commands
        client.moveByVelocityZAsync(
            vb[0],
            vb[1],
            ref_alt,
            duration,
            airsim.DrivetrainType.MaxDegreeOfFreedom,
            airsim.YawMode(True, vz),
        )
             
        # definition of episode completion (for now, I just wrote hacky done condition that ends each episode after two minutes)
        #if(time.time() > time.time() + 60*2):
        #    done = True
    
        img_counter = img_counter + 1
    # resets at the end of the episode
    
    client.reset()