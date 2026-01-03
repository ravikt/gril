"""Deploy trained GRIL model in Gazebo Harmonic for truck-following task.

Loads the gril.h5 model trained on AirSim data and deploys it in Gazebo
for autonomous yellow truck following.
"""

import argparse
import cv2
import numpy as np
import rclpy
import time
from pathlib import Path

try:
    import tensorflow as tf
except ImportError:
    print("TensorFlow not installed. Run: pip install tensorflow")
    exit(1)

from gazebo_utils import GazeboQuadrotorEnv


def preprocess_for_model(rgb: np.ndarray, depth: np.ndarray) -> tuple:
    """Preprocess sensor data for GRIL model inference.
    
    Args:
        rgb: RGB image from Gazebo (224x224x3)
        depth: Depth image from Gazebo (224x224x3)
        
    Returns:
        Tuple of preprocessed (rgb, depth) ready for model input
    """
    # Normalize to [0, 1] range (match training data)
    rgb_norm = rgb.astype(np.float32) / 255.0
    depth_norm = depth.astype(np.float32) / 255.0
    
    # Add batch dimension
    rgb_batch = np.expand_dims(rgb_norm, axis=0)
    depth_batch = np.expand_dims(depth_norm, axis=0)
    
    return rgb_batch, depth_batch


def run_gril_rollout(
    model_path: str,
    episodes: int = 5,
    max_steps: int = 200,
    namespace: str = 'quadrotor',
    save_data: bool = True,
    scale_constant: float = 10.0,
    visualization: bool = True
):
    """Execute GRIL model rollout for truck following in Gazebo.
    
    Args:
        model_path: Path to trained gril.h5 model
        episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        namespace: ROS2 namespace for quadrotor
        save_data: Whether to save trajectory data
        scale_constant: Scaling for velocity conversion
        visualization: Display predicted gaze overlays
    """
    # Load trained model
    print(f"\n{'='*60}")
    print(f"Loading GRIL model from: {model_path}")
    print('='*60)
    
    if not Path(model_path).exists():
        print(f"ERROR: Model file not found: {model_path}")
        return
    
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print("✓ Model loaded successfully!")
        model.summary()
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return
    
    # Initialize ROS2 and Gazebo environment
    print("\nInitializing Gazebo Harmonic environment...")
    rclpy.init()
    env = GazeboQuadrotorEnv(namespace=namespace, verbose=True)
    
    # Connect and prepare for flight
    env.connectQuadrotor()
    env.enableAPI(True)
    env.armQuadrotor()
    env.takeOff(altitude=2.0)
    
    print("Waiting for takeoff...")
    time.sleep(3.0)
    env.hover()
    print("✓ Ready for autonomous flight!\n")
    
    # Storage for trajectory data
    all_trajectories = []
    
    # Run episodes
    for episode in range(episodes):
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{episodes} - Following Yellow Truck")
        print('='*60)
        
        # Reset position
        env.teleportRelativeQuadrotor(0, 0, 0, 0)
        time.sleep(1.0)
        
        episode_data = {
            'rgb_images': [],
            'depth_images': [],
            'predicted_actions': [],
            'predicted_gazes': [],
            'timestamps': []
        }
        
        success_steps = 0
        
        for step in range(max_steps):
            step_start = time.time()
            
            # Get sensor data from Gazebo
            rgb = env.getRGBImage()
            depth = env.getDepthImage()
            
            if rgb is None or depth is None:
                print(f"  ⚠ Step {step}: No sensor data, skipping...")
                continue
            
            # Preprocess for model
            rgb_input, depth_input = preprocess_for_model(rgb, depth)
            
            # Run GRIL model inference
            try:
                predictions = model.predict(
                    [rgb_input, depth_input], 
                    verbose=0
                )
                action_pred, gaze_pred = predictions
                
                # Extract action commands [roll, pitch, throttle, yaw]
                roll, pitch, throttle, yaw = action_pred[0]
                gaze_x, gaze_y = gaze_pred[0]
                
            except Exception as e:
                print(f"  ✗ Step {step}: Prediction failed - {e}")
                continue
            
            # Convert model outputs to control commands
            vx, vy, vz, ref_alt = env.angularRatesToLinearVelocity(
                pitch, roll, yaw, throttle, scale_constant
            )
            
            # Transform to body frame
            vb = env.inertialToBodyFrame(yaw, vx, vy)
            
            # Send control command to quadrotor
            env.controlQuadrotor(vb, vz, ref_alt, duration=0.1)
            
            # Visualization: overlay gaze on image
            if visualization and step % 10 == 0:
                vis_img = rgb.copy()
                # Convert normalized gaze coords to pixel coords
                gaze_px = int(gaze_x * 224)
                gaze_py = int(gaze_y * 224)
                cv2.circle(vis_img, (gaze_px, gaze_py), 5, (0, 255, 0), -1)
                cv2.putText(
                    vis_img, f"Step {step}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                )
                cv2.imshow('GRIL Prediction', vis_img)
                cv2.waitKey(1)
            
            # Store data
            if save_data:
                episode_data['rgb_images'].append(rgb)
                episode_data['depth_images'].append(depth)
                episode_data['predicted_actions'].append(action_pred[0])
                episode_data['predicted_gazes'].append(gaze_pred[0])
                episode_data['timestamps'].append(time.time())
            
            # Process ROS callbacks
            rclpy.spin_once(env, timeout_sec=0.01)
            
            # Check for collision
            if env.hasCollided():
                print(f"  ✗ Collision detected at step {step}!")
                break
            
            success_steps += 1
            
            # Progress output
            if step % 20 == 0:
                print(f"  Step {step}/{max_steps} | "
                      f"Action: [{roll:.2f}, {pitch:.2f}, {throttle:.2f}, {yaw:.2f}] | "
                      f"Gaze: ({gaze_x:.2f}, {gaze_y:.2f})")
            
            # Maintain control frequency
            elapsed = time.time() - step_start
            if elapsed < 0.1:
                time.sleep(0.1 - elapsed)
        
        print(f"\n✓ Episode {episode + 1} completed: {success_steps}/{max_steps} steps")
        
        all_trajectories.append(episode_data)
        
        # Return to hover between episodes
        env.hover()
        time.sleep(2.0)
    
    # Save trajectory data
    if save_data and len(all_trajectories) > 0:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_path = f'gril_gazebo_rollout_{timestamp}.npz'
        print(f"\nSaving trajectory data to: {save_path}")
        
        np.savez_compressed(
            save_path,
            trajectories=all_trajectories,
            model_path=model_path,
            episodes=episodes,
            scale_constant=scale_constant
        )
        print(f"✓ Data saved successfully!")
    
    # Cleanup
    print("\n" + "="*60)
    print("Rollout complete. Shutting down...")
    print("="*60)
    
    if visualization:
        cv2.destroyAllWindows()
    
    env.hover()
    time.sleep(1.0)
    rclpy.shutdown()


def main():
    """Command-line interface for GRIL Gazebo deployment."""
    parser = argparse.ArgumentParser(
        description='Deploy trained GRIL model in Gazebo Harmonic for truck following',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '-m', '--model',
        type=str,
        default='gril.h5',
        help='Path to trained GRIL model'
    )
    parser.add_argument(
        '-e', '--episodes',
        type=int,
        default=5,
        help='Number of episodes'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=200,
        help='Maximum steps per episode'
    )
    parser.add_argument(
        '-ns', '--namespace',
        type=str,
        default='quadrotor',
        help='ROS2 namespace for quadrotor'
    )
    parser.add_argument(
        '-sc', '--scale',
        type=float,
        default=10.0,
        help='Velocity scaling constant'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save trajectory data'
    )
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Disable visualization'
    )
    
    args = parser.parse_args()
    
    run_gril_rollout(
        model_path=args.model,
        episodes=args.episodes,
        max_steps=args.max_steps,
        namespace=args.namespace,
        save_data=not args.no_save,
        scale_constant=args.scale,
        visualization=not args.no_viz
    )


if __name__ == "__main__":
    main()
