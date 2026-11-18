# Mecanum kinematics: conventions and derivation

This document derives every formula the library uses. Each numbered result is
checked by a test named in the text, so the write-up and the code cannot drift
apart silently.

## 1. Frames and sign conventions

The body frame follows ROS REP-103: **x forward, y left, z up**, and yaw `wz`
is positive counter-clockwise when seen from above.

```
                     x  (forward, +vx)
                     ^
                     |
          FL         |         FR
        +-----+      |      +-----+
        | \ \ |------+------| / / |      Lines inside each wheel show the
        | \ \ |      |      | / / |      roller axes AS SEEN FROM ABOVE
        +-----+      |      +-----+      (the top rollers). They form an "X".
           |         |         |
 y <-------|---------o---------|--       o = robot centre
 (left,    |    (+wz is CCW)   |         lx = centre to axle, along x
  +vy)  +-----+             +-----+      ly = centre to wheel, along y
        | / / |-------------| \ \ |
        | / / |             | \ \ |
        +-----+             +-----+
          RL                   RR

        |<------- 2 * ly ------->|       wheelbase = 2 * lx, track = 2 * ly
```

Wheel positions in the body frame:

| Wheel | x     | y     | top roller (drawn above) | ground roller axis angle `gamma` |
|-------|-------|-------|--------------------------|----------------------------------|
| FL    | `+lx` | `+ly` | `\`                      | `-alpha`                         |
| FR    | `+lx` | `-ly` | `/`                      | `+alpha`                         |
| RL    | `-lx` | `+ly` | `/`                      | `+alpha`                         |
| RR    | `-lx` | `-ly` | `\`                      | `-alpha`                         |

`alpha` is the roller angle (45 degrees for ordinary mecanum wheels), and
`gamma` is measured from the body x axis, counter-clockwise. The roller that
touches the ground is on the opposite side of the wheel from the one you see
from above, so its axis is mirrored: the "X" seen from above is an "O" on the
floor. This is the standard arrangement, and the only one supported. (The
other arrangement, "O" from above, cannot rotate properly: see section 4.)

**Wheel sign:** a positive wheel angular velocity `w_i` is the direction that
drives the robot forward. All four wheels positive means straight ahead. On a
real robot the motors on one side are mounted mirrored, so your motor driver
has to take care of that sign; this library has no per-wheel inversion flag.

**Wheel order:** the canonical order is `FL, FR, RL, RR`. `wheel_order` in the
config only permutes the rows of the matrices below, so that arrays match
whatever order your motor driver uses.

## 2. Rolling constraint of one wheel

Let the body twist be `(vx, vy, wz)`. The point of wheel `i` that touches the
ground, at body position `(x_i, y_i)`, moves with the rigid body at

```
c_i = ( vx - wz * y_i ,  vy + wz * x_i )
```

The wheel hub turns about the body y axis, so by itself it gives the contact
point a velocity `r * w_i` along x. The ground roller spins freely about its
own axis `a_i = (cos gamma_i, sin gamma_i)`, which means it can absorb any
velocity perpendicular to `a_i` but none along it. No slip along the roller
axis therefore requires the body velocity and the wheel-driven velocity to
have the same component along `a_i`:

```
a_i . c_i  =  a_i . (r * w_i, 0)

cos(gamma_i) * (vx - wz * y_i) + sin(gamma_i) * (vy + wz * x_i) = r * w_i * cos(gamma_i)
```

Dividing by `cos(gamma_i)`, with `tan(gamma_i) = s_i * tan(alpha)` and
`s = (-1, +1, +1, -1)` for `FL, FR, RL, RR`:

```
r * w_i = vx - wz * y_i + s_i * tan(alpha) * (vy + wz * x_i)             (1)
```

`alpha` must lie strictly between 0 and 90 degrees. At 0 the rollers turn the
wheel into an omni wheel with all four pointing the same way, and sideways
motion is uncontrollable; at 90 degrees `cos(gamma) = 0` and the wheel
cannot push at all. `RobotConfig` rejects both.

`test_general_roller_angle_against_first_principles` evaluates the constraint
above wheel by wheel, with a 30 degree roller, and compares it with the
matrix used by the library.

## 3. Inverse kinematics

Substituting the wheel positions into (1), with `t = tan(alpha)` and the
**rotation arm** `k = ly + lx * t`:

```
        [ w_FL ]         [ 1   -t   -k ]   [ vx ]
        [ w_FR ]   1     [ 1   +t   +k ]   [ vy ]
    w = [ w_RL ] = --- * [ 1   +t   -k ] * [ wz ]  =  J * v              (2)
        [ w_RR ]    r    [ 1   -t   +k ]
```

For 45 degree rollers `t = 1` and `k = lx + ly`, which gives the familiar

```
w_FL = (vx - vy - (lx + ly) * wz) / r
w_FR = (vx + vy + (lx + ly) * wz) / r
w_RL = (vx + vy - (lx + ly) * wz) / r
w_RR = (vx - vy + (lx + ly) * wz) / r
```

Sanity checks (`tests/test_kinematics.py`):

| Motion                 | Wheel speeds `FL, FR, RL, RR`           | Test |
|------------------------|------------------------------------------|------|
| forward `vx`           | `vx/r` on all four                       | `test_pure_forward_gives_equal_wheel_speeds_v_over_r` |
| strafe left `vy`       | `-vy/r, +vy/r, +vy/r, -vy/r`             | `test_pure_strafe_left_sign_pattern` |
| rotate CCW `wz`        | `-k*wz/r, +k*wz/r, -k*wz/r, +k*wz/r`     | `test_pure_rotation_ccw` |

## 4. Forward kinematics and the slip residual

Four wheel speeds determine three body velocities, so the system is
over-determined. The library takes the least-squares solution

```
v = J+ * w,      J+ = (J^T J)^-1 J^T
```

The three columns of `J` are mutually orthogonal (`J^T J = (4 / r^2) *
diag(1, t^2, k^2)`), so the pseudo-inverse has a closed form:

```
            [  1      1      1      1   ]
     r      [ -1/t   +1/t   +1/t   -1/t ]
J+ = --- *  [ -1/k   +1/k   -1/k   +1/k ]                                (3)
     4
```

The code computes `J+` numerically with `numpy.linalg.pinv`;
`test_jacobian_matches_closed_form` compares it with (3).

(With the unsupported "O from above" arrangement the yaw column becomes
`ly - lx * t`, which is zero for a square 45 degree robot: `J` loses rank and
the robot cannot be commanded to rotate. That is why only "X from above" is
implemented.)

**Residual.** The fourth dimension of wheel-speed space, orthogonal to all
columns of `J`, is spanned by the unit vector

```
n = (1, 1, -1, -1) / 2        (FL, FR, RL, RR)
```

i.e. front wheels forward and rear wheels backward. No rigid motion without
slip can produce this pattern: the rollers just grind against the floor. The
least-squares residual is exactly the component of `w` along `n`:

```
e = w - J * J+ * w = (n . w) * n,      n . w = (w_FL + w_FR - w_RL - w_RR) / 2      (4)
```

`ForwardResult` reports it three ways:

- `residual`: the vector `e` in rad/s, in `wheel_order`;
- `slip_rms = |e| / 2`: its RMS over the four wheels, in rad/s;
- `slip_ratio = |e| / |w|`: unitless, 0 for consistent wheel speeds and 1 when
  the wheels produce no rigid motion at all.

A non-zero residual proves that at least one wheel slipped or was measured
wrongly. A zero residual does **not** prove the absence of slip: slip that
happens to look like rigid motion (for example all four wheels spinning on
ice) is invisible to this metric. Worked example in
`test_slip_residual_for_one_fast_wheel`.

## 5. Saturation

If any wheel target exceeds `max_wheel_speed`, all four are multiplied by

```
scale = max_wheel_speed / max_i |w_i|                                    (5)
```

Because (2) is linear, the twist that is actually executed is `scale * v`:
same direction and same curvature, lower speed. Clipping each wheel on its own
would change the ratio between wheels and therefore bend the path
(`test_saturation_scales_uniformly_and_preserves_direction`,
`test_saturation_slows_the_robot_without_bending_the_path`).

## 6. Odometry

### Encoders to wheel angles

```
dphi_i = 2 * pi * dticks_i / (ticks_per_rev * gear_ratio)
```

`ticks_per_rev` is counted on the encoder shaft, as seen by your decoder (so
it already includes the factor 4 if you count every quadrature edge), and
`gear_ratio` is encoder-shaft turns per wheel turn.

### Wheel angles to body displacement

Integrating (2) over one sample interval, with the twist assumed constant in
the body frame during that interval:

```
(dx, dy, dtheta) = J+ * dphi
```

### Exact arc integration on SE(2)

A constant body twist traces a circular arc (or a straight line when
`dtheta = 0`), also when the robot is strafing while it turns. The pose
reached is given by the exponential map of SE(2):

```
[ px ]   [ sin(dtheta)/dtheta        -(1 - cos(dtheta))/dtheta ] [ dx ]
[ py ] = [ (1 - cos(dtheta))/dtheta    sin(dtheta)/dtheta      ] [ dy ]  (6)
```

and the world pose `(x, y, theta)` is updated by composing with it:

```
x     <- x + cos(theta) * px - sin(theta) * py
y     <- y + sin(theta) * px + cos(theta) * py
theta <- theta + dtheta
```

For `|dtheta| < 1e-6` the two fractions are replaced by their Taylor series
`1 - dtheta^2/6` and `dtheta/2 - dtheta^3/24` to avoid 0/0.

A plain Euler update would use `(px, py) = (dx, dy)`, i.e. it would drive the
whole step along the old heading. `test_arc_integration_matches_known_circle_with_coarse_steps`
drives a 0.5 m circle in only twelve 30 degree steps and lands on the circle
to 1e-12 m at every step; one Euler step of that size is already 68 mm off.
`theta` is not wrapped; use `wrap_angle` if you need `(-pi, pi]`.

What "exact" means here: exact for a twist that is constant in the body frame
between two encoder samples. If the twist changes within a sample, there is
still a discretisation error, as with any integrator.

### Covariance growth model

The model is deliberately simple. Each wheel's angle increment is assumed to
carry independent, zero-mean noise with variance proportional to how far the
wheel rolled:

```
var(dphi_i) = slip_noise^2 * |dphi_i|          Q = diag(var(dphi_i))     (7)
```

`slip_noise` has units of rad^0.5. Variance proportional to distance makes the
accumulated wheel error a random walk whose size depends on the distance
driven and not on the sampling rate
(`test_covariance_is_independent_of_sampling_rate_on_a_line`).

The 3x3 covariance `P` of `(x, y, theta)` is propagated to first order:

```
P <- F P F^T + G Q G^T

    [ 1  0  -wy ]
F = [ 0  1   wx ]          (wx, wy) = this step's displacement in the world frame
    [ 0  0   1  ]

G = Rz(theta + dtheta / 2) * J+
```

`F` carries existing heading uncertainty into position (a heading error of
`e` rad moves a step of length `d` sideways by `e * d`), and `G` maps wheel
noise into the world frame at the mid-step heading.

For a single straight step from `P = 0` the result is diagonal and can be
written down by hand, which is what
`test_covariance_of_straight_line_matches_closed_form` checks:

```
var(x) = var(y) = (r^2 / 4) * slip_noise^2 * dphi
var(theta)      = var(x) / (lx + ly)^2
```

`test_predicted_covariance_matches_monte_carlo_spread` runs the simulator 300
times with different seeds and checks that the predicted standard deviations
match the observed scatter of the odometry error to within 20%.

**Limits of the model.** It describes random slip only. It does not include:

- systematic errors (wrong wheel radius, wrong `lx`/`ly`, unequal wheels),
  which in practice dominate mecanum odometry and grow linearly, not as a
  random walk;
- encoder quantisation;
- correlation between wheels, or more slip when strafing than when rolling
  forward, which real mecanum wheels show.

`slip_noise` has to be tuned against your own robot. The library ships no
value that has been measured on hardware.

## 7. Simulator model

`simulate()` advances in fixed steps of `dt`. In each step:

1. The commanded twist is the time average of the trajectory over the step,
   so a segment boundary inside a step is weighted correctly.
2. Wheel targets come from (2) and (5).
3. Each wheel follows its target through a first-order lag,
   `dw/dt = (target - w) / tau`, discretised exactly for a target held over
   the step: `w <- w + (1 - exp(-dt / tau)) * (target - w)`. `tau = 0` means
   ideal tracking.
4. Wheel angles advance by `w * dt`. The encoders report the cumulative angle
   in counts, floored to whole counts if quantisation is on, and the odometry
   receives count differences, as it would on a robot.
5. Ground truth: each wheel's effective ground motion is its rotation plus a
   Gaussian slip angle drawn with the variance of (7); the body displacement
   is `J+` applied to those perturbed increments, integrated with (6).

Everything random comes from one `numpy.random.default_rng(seed)`, so a run
is repeatable (`test_same_seed_is_bit_identical_and_seeds_differ`).

This is a kinematic simulator. There are no masses, no friction model, no
motor torque or current limits and no wheel-ground contact dynamics. Slip is a
noise term, not physics. It is useful for testing odometry code, tuning
covariance parameters and trying command sequences; it does not predict how a
specific robot will behave on a specific floor.
