#!/usr/bin/env python3
"""
Leader-to-ROS2 Bridge - Lightweight bridge for Isaac Sim Virtual Follower

Reads joint positions from leader arm only and publishes to ROS2 topics,
allowing Isaac Sim to act as a virtual follower robot.

No physical follower required - Isaac Sim mirrors the leader movements.

Usage:
    python3 leader_to_ros2_bridge.py \
        --teleop.type=so101_leader \
        --teleop.port=/dev/ttyACM0 \
        --teleop.id=leader_asa \
        --fps=60
"""

import logging
import math
import time
from dataclasses import asdict, dataclass

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from lerobot.configs import parser
from lerobot.teleoperators import (
    TeleoperatorConfig,
    bi_so100_leader,
    gamepad,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    so100_leader,
    so101_leader,
)
from lerobot.utils.import_utils import register_third_party_devices
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import init_logging


# Motor names from LeRobot match Isaac Sim USD joint names exactly
# No mapping needed - use LeRobot names directly:
# shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper


class LeaderROS2Bridge(Node):
    """ROS2 node that publishes leader arm joint states for Isaac Sim."""
    
    def __init__(self):
        super().__init__('leader_ros2_bridge')
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.isaac_pub = self.create_publisher(JointState, '/isaac_joint_command', 10)
        self.get_logger().info('Leader-to-ROS2 Bridge initialized')
        self.get_logger().info('Publishing leader joint states to /joint_states and /isaac_joint_command')
        self.get_logger().info('Isaac Sim will act as virtual follower')
    
    def publish_joint_states(self, action: dict):
        """Convert leader action to ROS2 JointState message and publish."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        
        # Extract joint positions from leader action and convert to radians
        # Matches behavior of joint_state_reader.py
        for motor_name in ["shoulder_pan", "shoulder_lift", "elbow_flex", 
                          "wrist_flex", "wrist_roll", "gripper"]:
            key = f"{motor_name}.pos"
            if key in action:
                msg.name.append(motor_name)
                # Convert from degrees to radians (matching joint_state_reader.py)
                degrees = float(action[key])
                radians = math.radians(degrees)
                msg.position.append(radians)
        
        # Add empty velocity and effort arrays (required by Isaac Sim)
        msg.velocity = [0.0] * len(msg.position)
        msg.effort = [0.0] * len(msg.position)
        
        # Debug: Log occasionally to verify correct publishing
        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 0
        self._debug_counter += 1
        
        if self._debug_counter % 300 == 1:  # Every 5 seconds at 60Hz
            pos_str = ', '.join([f'{p:.3f}' for p in msg.position])
            self.get_logger().info(f"Publishing positions: [{pos_str}]")
            self.get_logger().info(f"All positions unique: {len(set(msg.position)) == len(msg.position)}")
        
        # Publish to both topics
        self.joint_pub.publish(msg)
        self.isaac_pub.publish(msg)


@dataclass
class LeaderROS2BridgeConfig:
    """Configuration for Leader-to-ROS2 Bridge."""
    teleop: TeleoperatorConfig
    fps: int = 60
    duration_s: float | None = None


def leader_publish_loop(
    teleop,
    ros2_bridge: LeaderROS2Bridge,
    fps: int,
    duration: float | None = None,
):
    """
    Main loop that reads leader positions and publishes to ROS2.
    
    This is a simple read-and-publish loop with no robot control,
    allowing Isaac Sim to act as the virtual follower.
    """
    start = time.perf_counter()
    iteration = 0

    while True:
        loop_start = time.perf_counter()

        # Read leader arm positions
        action = teleop.get_action()
        
        # Publish to ROS2
        ros2_bridge.publish_joint_states(action)
        
        # Spin ROS2 callbacks
        rclpy.spin_once(ros2_bridge, timeout_sec=0)

        iteration += 1
        
        # Status update every 5 seconds
        if iteration % (fps * 5) == 0:
            elapsed = time.perf_counter() - start
            ros2_bridge.get_logger().info(
                f'Running for {elapsed:.1f}s - Publishing at {fps}Hz'
            )

        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)
        loop_s = time.perf_counter() - loop_start
        print(f"\rtime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)", end='', flush=True)

        if duration is not None and time.perf_counter() - start >= duration:
            return


@parser.wrap()
def main(cfg: LeaderROS2BridgeConfig):
    """Main entry point for Leader-to-ROS2 Bridge."""
    import os
    
    init_logging()
    logging.info("Leader-to-ROS2 Bridge Configuration:")
    logging.info(f"  Teleop: {cfg.teleop.type}")
    logging.info(f"  Port: {cfg.teleop.port}")
    logging.info(f"  ID: {cfg.teleop.id}")
    logging.info(f"  FPS: {cfg.fps}")
    logging.info(f"  ROS_DOMAIN_ID: {os.environ.get('ROS_DOMAIN_ID', '0')}")
    
    # Initialize ROS2
    rclpy.init()
    ros2_bridge = LeaderROS2Bridge()
    
    # Initialize leader device only
    teleop = make_teleoperator_from_config(cfg.teleop)
    teleop.connect()

    try:
        leader_publish_loop(
            teleop=teleop,
            ros2_bridge=ros2_bridge,
            fps=cfg.fps,
            duration=cfg.duration_s,
        )
    except KeyboardInterrupt:
        print("\nShutting down Leader-to-ROS2 Bridge...")
    finally:
        teleop.disconnect()
        ros2_bridge.destroy_node()
        rclpy.shutdown()
        logging.info("Leader-to-ROS2 Bridge shutdown complete")


if __name__ == "__main__":
    register_third_party_devices()
    main()

