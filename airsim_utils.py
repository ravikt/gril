import airsim
import math
import numpy as np
import cv2
class AirSimEnv():
    
    def __init__(self):
        self.client = airsim.MultirotorClient()
    
    def connectQuadrotor(self) -> None:
        self.client.confirmConnection()

    def enableAPI(self, is_enable: bool) -> None:
        self.client.enableApiControl(is_enable)

    def reset(self) -> None:
        self.client.reset()

    def armQuadrotor(self) -> None:
        self.client.armDisarm(True)

    def takeOff(self) -> None:
        self.client.takeoffAsync().join()

    def hover(self) -> None:
        self.client.hoverAsync().join()

    def hasCollided(self):
        return 

    def getRGBImage(self) -> np.ndarray:
        # retrieve single RGB image from the camera
        response = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])[0]
        # numpy array
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        # for Unreal 4.25
        img_rgb = img1d.reshape(response.height, response.width, 3)

        return img_rgb

    def getDepthImage(self):
        
        response = self.client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.DepthVis, True)])[0]

        dp1d = np.array(response.image_data_float, dtype=np.float32)
        dp = dp1d.reshape(response.height, response.width)
        img_depth = np.array(np.abs(1-dp) * 255, dtype=np.uint8)

        return img_depth
 
    def saveImage(self, filename: str, image: np.ndarray) -> None:
        cv2.imwrite(filename, image)
        
    def angularRatesToLinearVelocity(self, pitch, roll, yaw, throttle, sc) -> tuple:
        vx = sc / 1.5 * pitch #1
        vy = sc / 1.5 * roll #0
        vz = 10 * sc * yaw #3
        self.state = self.client.getMultirotorState()
        ref_alt = self.state.kinematics_estimated.position.z_val + sc / 2 * throttle
        return (vx, vy, vz, ref_alt)

    def inertialToBodyFrame(self, yaw, vx, vy):
        C = np.zeros((2, 2))
        C[0, 0] = np.cos(yaw)
        C[0, 1] = -np.sin(yaw)
        C[1, 0] = -C[0, 1]
        C[1, 1] = C[0, 0]
        return C.dot(np.array([vx, vy]))

    def controlQuadrotor(self, vb, vz, ref_alt, duration):
        self.client.moveByVelocityZAsync(
            vb[0],
            vb[1],
            ref_alt,
            duration,
            airsim.DrivetrainType.MaxDegreeOfFreedom,
            airsim.YawMode(True, vz),
        )

    def toEulerianAngle(q):
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
    
    def teleportRelativeQuadrotor(self, x, y, z, yaw):
        pose = self.client.simGetVehiclePose()
        pose.position.x_val += x
        pose.position.y_val += y
        pose.position.z_val += z
        pose.orientation.z_val += yaw
        self.client.simSetVehiclePose(pose, True, "SimpleFlight")

