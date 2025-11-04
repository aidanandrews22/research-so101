#!/usr/bin/env python3
"""
LeRobot ROS2 Bridge - Middleware for Isaac Sim Integration

This script runs LeRobot teleoperation while simultaneously publishing
follower robot joint states to ROS2 topics for Isaac Sim visualization.

Solves port contention by having only LeRobot access the hardware directly.

Usage:
    python3 lerobot_ros2_bridge.py \
        --robot.type=so101_follower \
        --robot.port=/dev/ttyACM1 \
        --robot.id=follower_asa \
        --teleop.type=so101_leader \
        --teleop.port=/dev/ttyACM0 \
        --teleop.id=leader_asa \
        --fps=60
"""

import logging
import math
import time
from dataclasses import asdict, dataclass
from pprint import pformat

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import rerun as rr

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.configs import parser
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.robots import (
    Robot,
    RobotConfig,
    bi_so100_follower,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.teleoperators import (
    Teleoperator,
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
from lerobot.utils.utils import init_logging, move_cursor_up
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


# Motor names from LeRobot match Isaac Sim USD joint names exactly
# No mapping needed - use LeRobot names directly:
# shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper


class LeRobotROS2Bridge(Node):
    """ROS2 node that publishes follower robot joint states."""
    
    def __init__(self):
        super().__init__('lerobot_ros2_bridge')
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.isaac_pub = self.create_publisher(JointState, '/isaac_joint_command', 10)
        self.get_logger().info('LeRobot ROS2 Bridge initialized')
        self.get_logger().info('Publishing follower joint states to /joint_states and /isaac_joint_command')
    
    def publish_joint_states(self, observation: dict):
        """Convert LeRobot observation to ROS2 JointState message and publish."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        
        # Extract joint positions from observation and convert to radians
        # Matches behavior of joint_state_reader.py
        for motor_name in ["shoulder_pan", "shoulder_lift", "elbow_flex", 
                          "wrist_flex", "wrist_roll", "gripper"]:
            key = f"{motor_name}.pos"
            if key in observation:
                msg.name.append(motor_name)
                # Convert from degrees to radians (matching joint_state_reader.py)
                degrees = float(observation[key])
                radians = math.radians(degrees)
                msg.position.append(radians)
        
        # Add empty velocity and effort arrays (required by Isaac Sim)
        msg.velocity = [0.0] * len(msg.position)
        msg.effort = [0.0] * len(msg.position)
        
        # Publish to both topics
        self.joint_pub.publish(msg)
        self.isaac_pub.publish(msg)


@dataclass
class LeRobotROS2BridgeConfig:
    """Configuration for LeRobot ROS2 Bridge."""
    teleop: TeleoperatorConfig
    robot: RobotConfig
    fps: int = 60
    teleop_time_s: float | None = None
    display_data: bool = False


def teleop_loop_with_ros2(
    teleop: Teleoperator,
    robot: Robot,
    ros2_bridge: LeRobotROS2Bridge,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    display_data: bool = False,
    duration: float | None = None,
):
    """
    Teleoperation loop that publishes follower observations to ROS2.
    
    This function runs the standard LeRobot teleoperation loop while simultaneously
    publishing the follower robot's joint states to ROS2 for Isaac Sim integration.
    """
    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()

    while True:
        loop_start = time.perf_counter()

        # Get robot observation (follower state)
        obs = robot.get_observation()
        
        # Publish follower state to ROS2
        ros2_bridge.publish_joint_states(obs)
        
        # Spin ROS2 callbacks
        rclpy.spin_once(ros2_bridge, timeout_sec=0)

        # Get teleop action (leader state)
        raw_action = teleop.get_action()

        # Process teleop action through pipeline
        teleop_action = teleop_action_processor((raw_action, obs))

        # Process action for robot through pipeline
        robot_action_to_send = robot_action_processor((teleop_action, obs))

        # Send processed action to robot
        _ = robot.send_action(robot_action_to_send)

        if display_data:
            # Process robot observation through pipeline
            obs_transition = robot_observation_processor(obs)

            log_rerun_data(
                observation=obs_transition,
                action=teleop_action,
            )

            print("\n" + "-" * (display_len + 10))
            print(f"{'NAME':<{display_len}} | {'NORM':>7}")
            # Display the final robot action that was sent
            for motor, value in robot_action_to_send.items():
                print(f"{motor:<{display_len}} | {value:>7.2f}")
            move_cursor_up(len(robot_action_to_send) + 5)

        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)
        loop_s = time.perf_counter() - loop_start
        print(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")

        if duration is not None and time.perf_counter() - start >= duration:
            return


@parser.wrap()
def main(cfg: LeRobotROS2BridgeConfig):
    """Main entry point for LeRobot ROS2 Bridge."""
    init_logging()
    logging.info(pformat(asdict(cfg)))
    
    # Initialize ROS2
    rclpy.init()
    ros2_bridge = LeRobotROS2Bridge()
    
    if cfg.display_data:
        init_rerun(session_name="lerobot_ros2_bridge")

    # Initialize LeRobot components
    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    teleop.connect()
    robot.connect()

    try:
        teleop_loop_with_ros2(
            teleop=teleop,
            robot=robot,
            ros2_bridge=ros2_bridge,
            fps=cfg.fps,
            display_data=cfg.display_data,
            duration=cfg.teleop_time_s,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        )
    except KeyboardInterrupt:
        print("\nShutting down LeRobot ROS2 Bridge...")
    finally:
        if cfg.display_data:
            rr.rerun_shutdown()
        teleop.disconnect()
        robot.disconnect()
        ros2_bridge.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    register_third_party_devices()
    main()

