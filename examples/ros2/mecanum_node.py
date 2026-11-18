#!/usr/bin/env python3
"""Example ROS 2 (rclpy) node wrapping mecanum-kinematics.

STATUS: EXAMPLE ONLY. This file is linted and byte-compiled in CI, but it is
NOT executed there (CI has no ROS installation) and it has NOT been run on a
robot or in a ROS 2 simulation by the author. Treat it as a starting point and
test it on your own system with the wheels off the ground first.

All of the logic lives in ``mecanum_kinematics.DriveBridge``, which has no ROS
dependency and is covered by the unit tests; this node only copies values
between ROS messages and that class.

Topics
    cmd_vel       (geometry_msgs/Twist, subscribed)   body velocity command
    wheel_ticks   (std_msgs/Int64MultiArray, subscribed, only if open_loop is
                  false) cumulative encoder counts in the config's wheel_order
    wheel_speeds  (std_msgs/Float64MultiArray, published) wheel speed targets
                  in rad/s, in the config's wheel_order
    odom          (nav_msgs/Odometry, published)

Parameters
    config        TOML path or bundled example name (default "generic-small")
    rate_hz       control loop rate (default 50.0)
    cmd_timeout   seconds before a stale cmd_vel is replaced by a stop (0.5)
    open_loop     if true (default), odometry integrates the commanded wheel
                  speeds; if false, it integrates wheel_ticks
    slip_noise    covariance growth parameter in rad^0.5 (default 0.02)
    odom_frame    header.frame_id of odom (default "odom")
    base_frame    child_frame_id of odom (default "base_link")

Not done here: no TF broadcast, no motor driver. Your hardware interface must
consume wheel_speeds and, for closed-loop odometry, publish wheel_ticks.

Run (after `pip install mecanum-kinematics` into the ROS Python environment):
    python3 mecanum_node.py --ros-args -p config:=/path/to/robot.toml
"""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import Twist as TwistMsg
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Int64MultiArray

from mecanum_kinematics import DriveBridge, OdomSample, Twist, load_config


class MecanumNode(Node):
    def __init__(self) -> None:
        super().__init__("mecanum_kinematics")
        self.declare_parameter("config", "generic-small")
        self.declare_parameter("rate_hz", 50.0)
        self.declare_parameter("cmd_timeout", 0.5)
        self.declare_parameter("open_loop", True)
        self.declare_parameter("slip_noise", 0.02)
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")

        config = load_config(str(self.get_parameter("config").value))
        self.bridge = DriveBridge(
            config,
            command_timeout=float(self.get_parameter("cmd_timeout").value),
            slip_noise=float(self.get_parameter("slip_noise").value),
        )
        self.open_loop = bool(self.get_parameter("open_loop").value)
        self.odom_frame = str(self.get_parameter("odom_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.period = 1.0 / float(self.get_parameter("rate_hz").value)
        self.last_tick_time: float | None = None

        self.wheel_pub = self.create_publisher(Float64MultiArray, "wheel_speeds", 10)
        self.odom_pub = self.create_publisher(Odometry, "odom", 10)
        self.create_subscription(TwistMsg, "cmd_vel", self.on_cmd_vel, 10)
        if not self.open_loop:
            self.create_subscription(Int64MultiArray, "wheel_ticks", self.on_wheel_ticks, 10)
        self.create_timer(self.period, self.on_timer)
        self.get_logger().info(
            f"mecanum_kinematics example node: config '{config.name}', "
            f"wheel order {list(config.wheel_order)}, open_loop={self.open_loop}"
        )

    def now_seconds(self) -> float:
        return float(self.get_clock().now().nanoseconds) * 1e-9

    def on_cmd_vel(self, msg: TwistMsg) -> None:
        self.bridge.set_command(
            Twist(float(msg.linear.x), float(msg.linear.y), float(msg.angular.z)),
            self.now_seconds(),
        )

    def on_timer(self) -> None:
        command = self.bridge.wheel_targets(self.now_seconds())
        out = Float64MultiArray()
        out.data = [float(v) for v in command.speeds]
        self.wheel_pub.publish(out)
        if self.open_loop:
            self.publish_odom(self.bridge.update_from_wheel_speeds(command.speeds, self.period))

    def on_wheel_ticks(self, msg: Int64MultiArray) -> None:
        if len(msg.data) != 4:
            self.get_logger().warning(f"wheel_ticks needs 4 values, got {len(msg.data)}")
            return
        now = self.now_seconds()
        dt = 0.0 if self.last_tick_time is None else max(0.0, now - self.last_tick_time)
        self.last_tick_time = now
        self.publish_odom(self.bridge.update_from_cumulative_ticks(list(msg.data), dt))

    def publish_odom(self, sample: OdomSample) -> None:
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.odom_frame
        msg.child_frame_id = self.base_frame
        msg.pose.pose.position.x = sample.pose.x
        msg.pose.pose.position.y = sample.pose.y
        qx, qy, qz, qw = sample.quaternion
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw
        msg.pose.covariance = sample.pose_covariance
        msg.twist.twist.linear.x = sample.twist.vx
        msg.twist.twist.linear.y = sample.twist.vy
        msg.twist.twist.angular.z = sample.twist.wz
        self.odom_pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = MecanumNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
