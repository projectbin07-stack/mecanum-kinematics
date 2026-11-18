# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-11-18

First release.

### Added

- `RobotConfig` loaded from TOML or a dict, with validation and three bundled
  illustrative example configs.
- Inverse kinematics, least-squares forward kinematics, slip residual and
  direction-preserving wheel speed saturation, for any roller angle between 0
  and 90 degrees and any wheel order.
- `MecanumOdometry`: encoder ticks to pose with exact SE(2) arc integration
  and a first-order covariance growth model.
- Deterministic simulator with first-order motor lag, optional encoder
  quantisation and seeded Gaussian wheel slip; line, strafe, square, circle,
  figure-eight and spin trajectories; drift report.
- `mecanum` CLI with `ik`, `fk`, `simulate`, `odom` and `configs` commands.
- Pure-Python SVG path plots, and an optional matplotlib figure.
- Encoder tick CSV reader and replay.
- `DriveBridge`, a middleware-free command watchdog and odometry sampler, and
  an example ROS 2 node built on it (not executed in CI).
- Derivation and conventions in `docs/kinematics.md`.

### Known limitations

- Not validated on hardware; the ROS 2 example has not been run.

[0.1.0]: https://github.com/projectbin07-stack/mecanum-kinematics/releases/tag/v0.1.0
