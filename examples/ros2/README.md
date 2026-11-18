# ROS 2 adapter example

`mecanum_node.py` is a thin `rclpy` node around `mecanum_kinematics.DriveBridge`.

**Status: not executed in CI and not tested on a robot.** CI only lints and
byte-compiles this file, because the CI runners have no ROS installation. The
author has not run it against ROS 2 Humble or any other distribution. The
logic it calls (`DriveBridge`: command watchdog, saturated inverse kinematics,
odometry samples, quaternion and covariance layout) is pure Python and is
covered by `tests/test_bridge.py`.

## What it does

| Direction | Topic          | Type                         | Meaning                                         |
|-----------|----------------|------------------------------|-------------------------------------------------|
| subscribe | `cmd_vel`      | `geometry_msgs/Twist`        | body velocity command (`linear.x/y`, `angular.z`) |
| subscribe | `wheel_ticks`  | `std_msgs/Int64MultiArray`   | cumulative encoder counts (only if `open_loop:=false`) |
| publish   | `wheel_speeds` | `std_msgs/Float64MultiArray` | wheel targets in rad/s                           |
| publish   | `odom`         | `nav_msgs/Odometry`          | pose, body twist, pose covariance                |

Wheel arrays use the `wheel_order` of the loaded config.

With `open_loop:=true` (the default) the node integrates its own commanded
wheel speeds. That needs no encoders but knows nothing about what the wheels
really did, so treat the result as a placeholder for wiring up the rest of a
stack, not as odometry you can navigate with.

## What it does not do

- It does not talk to motors. Something else must turn `wheel_speeds` into PWM.
- It does not broadcast the `odom -> base_link` transform.
- It is not packaged as an `ament_python` package; it is a single script.

## Trying it

```bash
# inside a sourced ROS 2 environment
pip install mecanum-kinematics   # or: pip install -e /path/to/this/repo
python3 mecanum_node.py --ros-args -p config:=/path/to/your_robot.toml

# in another terminal
ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.2}}"
ros2 topic echo /wheel_speeds
```

Lift the robot off the ground before the first run, and check that a positive
`linear.x` turns all four wheels forward. If one runs backwards, fix the motor
wiring or sign in your hardware interface; this library has no per-wheel
direction flag yet.
