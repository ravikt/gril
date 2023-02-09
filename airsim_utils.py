import airsim
import math

class AirSimEnv():
    
    def __init__():
        # ensure connection to Airsim Env
        # reset client?
        pass
    
    def connectQuadrotor(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

    def enableAPI(self, is_enable):
        self.client.enableApiControl(is_enable)

    def reset(self):
        self.client.reset()

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

