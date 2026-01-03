"""Gazebo Harmonic simulation environment utilities for quadrotor control.

ROS2 Jazzy + Gazebo Harmonic based interface for quadrotor truck-following task.
Provides compatibility layer for models trained on AirSim data.

Requirements:
    - ROS2 Jazzy
    - Gazebo Harmonic
    - gz-sim, gz-transport, gz-msgs
    - cv_bridge, sensor_msgs, geometry_msgs, nav_msgs
"""

import numpy as np
import cv2
import math
from typing import Optional, Tuple

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, CameraInfo
    from geometry_msgs.msg import Twist, PoseStamped, Quaternion
    from nav_msgs.msg import Odometry
    from std_srvs.srv import Empty
    from cv_bridge import CvBridge
except ImportError:
    print("Warning: ROS2 Jazzy packages not installed.")
    print("Install: sudo apt install ros-jazzy-desktop ros-jazzy-cv-bridge")


class GazeboQuadrotorEnv(Node):
    """ROS2 Jazzy + Gazebo Harmonic environment for quadrotor control.
    
    Designed for yellow truck following task, compatible with AirSim-trained models.
    Maintains API similarity to AirSimEnv for easy migration.
    """
    
    def __init__(self, namespace: str = "quadrotor", verbose: bool = True):
        """Initialize Gazebo quadrotor environment.
        
        Args:
            namespace: ROS namespace for the quadrotor
            verbose: Enable detailed logging
        """
        super().__init__('gazebo_quadrotor_env')
        
        self.namespace = namespace
        self.verbose = verbose
        self.bridge = CvBridge()
        
        # State variables
        self.rgb_image = None
        self.depth_image = None
        self.current_pose = None
        self.current_velocity = None
        self.is_armed = False
        
        # Camera parameters (match AirSim training data)
        self.target_width = 224
        self.target_height = 224
        
        # ROS2 subscribers
        self.rgb_sub = self.create_subscription(
            Image,
            f'/{namespace}/camera/rgb/image_raw',
            self._rgb_callback,
            10
        )
        
        self.depth_sub = self.create_subscription(
            Image,
            f'/{namespace}/camera/depth/image_raw',
            self._depth_callback,
            10
        )
        
        self.odom_sub = self.create_subscription(
            Odometry,
            f'/{namespace}/odom',
            self._odom_callback,
            10
        )
        
        # ROS2 publishers
        self.vel_pub = self.create_publisher(
            Twist,
            f'/{namespace}/cmd_vel',
            10
        )
        
        self.pose_pub = self.create_publisher(
            PoseStamped,
            f'/{namespace}/command/pose',
            10
        )
        
        # Service clients
        self.takeoff_client = self.create_client(Empty, f'/{namespace}/takeoff')
        
        if self.verbose:
            self.get_logger().info(
                f'Gazebo Harmonic environment initialized for {namespace}'
            )
    
    def _rgb_callback(self, msg: Image) -> None:
        """Callback for RGB camera images."""
        try:
            # Convert to BGR format (OpenCV/AirSim compatible)
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'RGB conversion failed: {e}')
    
    def _depth_callback(self, msg: Image) -> None:
        """Callback for depth camera images.
        
        Processes depth to match AirSim format for model compatibility.
        """
        try:
            # Get raw depth
            depth_float = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            
            # AirSim compatibility: normalize similar to AirSim DepthVis
            # AirSim depth is typically in meters, normalized to 0-255
            depth_normalized = cv2.normalize(
                depth_float, None, 0, 255, 
                cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            
            # Convert to 3-channel for model input (matches training data format)
            self.depth_image = cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2BGR)
            
        except Exception as e:
            self.get_logger().error(f'Depth conversion failed: {e}')
    
    def _odom_callback(self, msg: Odometry) -> None:
        """Callback for odometry updates."""
        self.current_pose = msg.pose.pose
        self.current_velocity = msg.twist.twist
    
    def connectQuadrotor(self) -> None:
        """Establish connection to Gazebo simulation.
        
        Maintains AirSim API naming for compatibility.
        """
        if self.verbose:
            self.get_logger().info('Connecting to Gazebo Harmonic...')
        
        timeout = 10.0
        start_time = self.get_clock().now()
        
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            
            if self.rgb_image is not None and self.current_pose is not None:
                if self.verbose:
                    self.get_logger().info('✓ Connected to Gazebo!')
                break
            
            elapsed = (self.get_clock().now() - start_time).nanoseconds / 1e9
            if elapsed > timeout:
                self.get_logger().warning(
                    'Timeout waiting for topics. Check if Gazebo is running with correct topic names.'
                )
                break
    
    def enableAPI(self, is_enable: bool) -> None:
        """Enable/disable API control (AirSim compatibility)."""
        self.is_armed = is_enable
        if self.verbose:
            self.get_logger().info(f'API control: {"enabled" if is_enable else "disabled"}')
    
    def reset(self) -> None:
        """Reset simulation state."""
        if self.verbose:
            self.get_logger().info('Reset requested')
        # TODO: Call Gazebo reset service
    
    def armQuadrotor(self) -> None:
        """Arm the quadrotor motors (AirSim compatibility)."""
        self.is_armed = True
        if self.verbose:
            self.get_logger().info('✓ Quadrotor armed')
    
    def takeOff(self, altitude: float = 2.0) -> None:
        """Command quadrotor to take off (AirSim API naming)."""
        if self.takeoff_client.wait_for_service(timeout_sec=1.0):
            request = Empty.Request()
            self.takeoff_client.call_async(request)
            if self.verbose:
                self.get_logger().info(f'Taking off to {altitude}m...')
        else:
            # Fallback: publish target pose
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = 'world'
            if self.current_pose:
                pose_msg.pose = self.current_pose
                pose_msg.pose.position.z = altitude
            self.pose_pub.publish(pose_msg)
    
    def hover(self) -> None:
        """Command quadrotor to hover."""
        twist = Twist()
        self.vel_pub.publish(twist)
        if self.verbose:
            self.get_logger().info('Hovering...')
    
    def hasCollided(self) -> bool:
        """Check for collision (requires collision plugin)."""
        # TODO: Implement via Gazebo collision sensor
        return False
    
    def getRGBImage(self) -> Optional[np.ndarray]:
        """Get RGB image (AirSim API naming).
        
        Returns:
            np.ndarray: RGB image in BGR format, resized to 224x224 for model input
        """
        rclpy.spin_once(self, timeout_sec=0.01)
        
        if self.rgb_image is None:
            return None
        
        # Resize to match training data dimensions
        resized = cv2.resize(
            self.rgb_image, 
            (self.target_width, self.target_height),
            interpolation=cv2.INTER_AREA
        )
        return resized
    
    def getDepthImage(self) -> Optional[np.ndarray]:
        """Get depth image (AirSim API naming).
        
        Returns:
            np.ndarray: Depth image in 3-channel format, resized to 224x224
        """
        rclpy.spin_once(self, timeout_sec=0.01)
        
        if self.depth_image is None:
            return None
        
        # Resize to match training data dimensions
        resized = cv2.resize(
            self.depth_image,
            (self.target_width, self.target_height),
            interpolation=cv2.INTER_AREA
        )
        return resized
    
    def saveImage(self, filename: str, image: np.ndarray) -> None:
        """Save image to disk (AirSim API)."""
        cv2.imwrite(filename, image)
        if self.verbose:
            self.get_logger().info(f'Saved: {filename}')
    
    def angularRatesToLinearVelocity(
        self, pitch: float, roll: float, yaw: float, throttle: float, sc: float
    ) -> Tuple[float, float, float, float]:
        """Convert angular rates to linear velocities (AirSim API naming).
        
        Maintains exact same conversion as original AirSim implementation
        for model compatibility.
        """
        vx = sc / 1.5 * pitch
        vy = sc / 1.5 * roll
        vz = 10 * sc * yaw
        
        ref_alt = -2.0
        if self.current_pose:
            ref_alt = self.current_pose.position.z + sc / 2 * throttle
        
        return (vx, vy, vz, ref_alt)
    
    def inertialToBodyFrame(self, yaw: float, vx: float, vy: float) -> np.ndarray:
        """Transform to body frame (AirSim API naming)."""
        C = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)]
        ])
        return C.dot(np.array([vx, vy]))
    
    def controlQuadrotor(
        self, vb: np.ndarray, vz: float, ref_alt: float, duration: float
    ) -> None:
        """Send velocity commands (AirSim API naming)."""
        twist = Twist()
        twist.linear.x = float(vb[0])
        twist.linear.y = float(vb[1])
        twist.linear.z = 0.0
        twist.angular.z = float(vz)
        
        self.vel_pub.publish(twist)
    
    @staticmethod
    def toEulerianAngle(q: Quaternion) -> Tuple[float, float, float]:
        """Convert quaternion to Euler (AirSim API naming)."""
        x, y, z, w = q.x, q.y, q.z, q.w
        ysqr = y * y
        
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + ysqr)
        roll = math.atan2(t0, t1)
        
        t2 = 2.0 * (w * y - z * x)
        t2 = max(-1.0, min(1.0, t2))
        pitch = math.asin(t2)
        
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (ysqr + z * z)
        yaw = math.atan2(t3, t4)
        
        return (pitch, roll, yaw)
    
    def teleportRelativeQuadrotor(
        self, x: float, y: float, z: float, yaw: float
    ) -> None:
        """Teleport quadrotor (AirSim API naming)."""
        if self.current_pose is None:
            return
        
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'world'
        
        pose_msg.pose.position.x = self.current_pose.position.x + x
        pose_msg.pose.position.y = self.current_pose.position.y + y
        pose_msg.pose.position.z = self.current_pose.position.z + z
        
        # Update yaw
        current_pitch, current_roll, current_yaw = self.toEulerianAngle(
            self.current_pose.orientation
        )
        new_yaw = current_yaw + yaw
        
        pose_msg.pose.orientation = self._euler_to_quaternion(
            current_roll, current_pitch, new_yaw
        )
        
        self.pose_pub.publish(pose_msg)
    
    @staticmethod
    def _euler_to_quaternion(roll: float, pitch: float, yaw: float) -> Quaternion:
        """Convert Euler to quaternion."""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        q = Quaternion()
        q.w = cr * cp * cy + sr * sp * sy
        q.x = sr * cp * cy - cr * sp * sy
        q.y = cr * sp * cy + sr * cp * sy
        q.z = cr * cp * sy - sr * sp * cy
        
        return q
