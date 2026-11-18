# Security policy

## Supported versions

Only the latest release receives fixes.

## Reporting a vulnerability

Please do not open a public issue for a security problem. Use the private
vulnerability reporting of this repository on GitHub (Security tab, "Report a
vulnerability"):
https://github.com/projectbin07-stack/mecanum-kinematics/security/advisories/new

Include what you found, how to reproduce it and the version or commit. This is
a spare-time project maintained by one person, so there is no guaranteed
response time.

## Scope

The library parses local TOML and CSV files and writes SVG, CSV and image
files to paths given by the user. It opens no network connections and
executes no input as code. Text placed in SVG output is XML-escaped.

## Safety note

This is not a safety-rated component. It has not been validated on hardware.
Do not rely on it as the only layer between a command source and motors:
keep hardware current limits, an emergency stop and a watchdog in the motor
controller itself.
