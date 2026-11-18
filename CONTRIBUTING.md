# Contributing

Bug reports, corrections to the math and pull requests are welcome.

## Setup

```bash
git clone https://github.com/projectbin07-stack/mecanum-kinematics.git
cd mecanum-kinematics
python -m venv .venv && . .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Checks

All four must pass; CI runs the same commands on Ubuntu and Windows with
Python 3.10 and 3.13.

```bash
ruff check .
ruff format --check .
mypy
pytest --cov
```

## Guidelines

- Keep the runtime dependency list at numpy (plus the `tomli` backport on
  Python 3.10). Anything else belongs in an optional extra.
- Every formula needs a test against a value you can derive by hand, with the
  derivation in a comment or in `docs/kinematics.md`. A test that only checks
  that code runs is not enough.
- If you change a sign convention or a formula, update `docs/kinematics.md`
  in the same pull request.
- Do not add robot configs that claim to describe a specific commercial
  product unless you measured it yourself and say so in the file.
- Be explicit about what was and was not tested on hardware. Results from a
  real robot are very welcome in issues; say which robot, floor and speed.
- Small, focused commits with imperative messages.

## Reporting a kinematics bug

The most useful report contains the config (TOML), the input (twist, wheel
speeds or tick CSV), what you got and what you expected, with the reasoning
for the expected value.
