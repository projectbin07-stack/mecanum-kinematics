"""Static checks of the ROS 2 example.

rclpy is not available in CI, so the node is never imported or run here. These
tests only make sure the file compiles and that everything it asks of this
library really exists. The behaviour it relies on is tested in test_bridge.py.
"""

from __future__ import annotations

import ast
from pathlib import Path

import mecanum_kinematics
from mecanum_kinematics import DriveBridge, OdomSample

NODE = Path(__file__).parent.parent / "examples" / "ros2" / "mecanum_node.py"


def _tree() -> ast.Module:
    return ast.parse(NODE.read_text(encoding="utf-8"), filename=str(NODE))


def test_example_compiles() -> None:
    compile(NODE.read_text(encoding="utf-8"), str(NODE), "exec")


def test_example_states_that_it_is_untested() -> None:
    doc = ast.get_docstring(_tree())
    assert doc is not None
    assert "NOT executed" in doc
    assert "NOT been run on a" in doc


def test_names_imported_from_the_library_exist() -> None:
    imported = [
        alias.name
        for node in ast.walk(_tree())
        if isinstance(node, ast.ImportFrom) and node.module == "mecanum_kinematics"
        for alias in node.names
    ]
    assert "DriveBridge" in imported
    for name in imported:
        assert hasattr(mecanum_kinematics, name), name


def test_bridge_and_sample_attributes_used_by_the_node_exist() -> None:
    bridge_calls: set[str] = set()
    sample_fields: set[str] = set()
    for node in ast.walk(_tree()):
        if not isinstance(node, ast.Attribute):
            continue
        owner = node.value
        if (
            isinstance(owner, ast.Attribute)
            and owner.attr == "bridge"
            and isinstance(owner.value, ast.Name)
            and owner.value.id == "self"
        ):
            bridge_calls.add(node.attr)
        if isinstance(owner, ast.Name) and owner.id == "sample":
            sample_fields.add(node.attr)
    assert bridge_calls == {
        "set_command",
        "wheel_targets",
        "update_from_wheel_speeds",
        "update_from_cumulative_ticks",
    }
    for name in bridge_calls:
        assert callable(getattr(DriveBridge, name))
    assert sample_fields == {"pose", "quaternion", "pose_covariance", "twist"}
    assert sample_fields <= set(OdomSample.__dataclass_fields__)
