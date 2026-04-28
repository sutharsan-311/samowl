#!/usr/bin/env python3
"""Nav mission node — drives the robot to a named object.

Subscribes to the object registry and a goal label topic. When a label
arrives, it looks up the object's approach pose and publishes it to Nav2's
/goal_pose topic. The robot navigates to the approach point, arriving
oriented toward the object's visible face.

Subscribed topics
-----------------
/samowl/object_registry  (std_msgs/String)  — live object map from object_registry_node
/samowl/nav_goal_label   (std_msgs/String)  — label to navigate to, e.g. "chair"

Published topics
----------------
/goal_pose  (geometry_msgs/PoseStamped)     — Nav2 navigation goal

Parameters
----------
map_frame      : str   — TF frame for the goal pose (default: map)
goal_tolerance : float — minimum approach distance override in metres; 0 = use hotspot value (default: 0.0)
"""

import json
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String


def _yaw_to_quaternion(yaw: float):
    """Convert a yaw angle (radians) to a geometry_msgs quaternion dict."""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return {'x': 0.0, 'y': 0.0, 'z': sy, 'w': cy}


class NavMissionNode(Node):
    def __init__(self):
        super().__init__('nav_mission_node')

        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('goal_tolerance', 0.0)

        self._map_frame = self.get_parameter('map_frame').get_parameter_value().string_value
        self._goal_tolerance = self.get_parameter('goal_tolerance').get_parameter_value().double_value

        self._registry: dict[str, dict] = {}

        self._registry_sub = self.create_subscription(
            String, '/samowl/object_registry', self._on_registry, 10)

        self._label_sub = self.create_subscription(
            String, '/samowl/nav_goal_label', self._on_label, 10)

        self._goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

        self.get_logger().info('Nav mission node ready')

    # ------------------------------------------------------------------ #

    def _on_registry(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f'Malformed registry JSON: {exc}')
            return

        self._registry = {}
        for obj in data.get('objects', []):
            label = obj.get('label', '')
            if not label:
                continue
            existing = self._registry.get(label)
            if existing is None or obj.get('confidence', 0.0) > existing.get('confidence', 0.0):
                self._registry[label] = obj

        self.get_logger().debug(
            f'Registry: {list(self._registry.keys())}')

    def _on_label(self, msg: String) -> None:
        label = msg.data.strip()
        if not label:
            return

        obj = self._registry.get(label)
        if obj is None:
            self.get_logger().warn(
                f'Object "{label}" not in registry. Known: {list(self._registry.keys())}')
            return

        approach = obj.get('approach')
        if not approach:
            self.get_logger().warn(f'Object "{label}" has no approach pose — geometry not computed')
            return

        pos = approach.get('position', [0.0, 0.0, 0.0])
        yaw = float(approach.get('yaw_rad', 0.0))
        q = _yaw_to_quaternion(yaw)

        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = obj.get('map_frame', self._map_frame)
        goal.pose.position.x = float(pos[0])
        goal.pose.position.y = float(pos[1])
        goal.pose.position.z = 0.0
        goal.pose.orientation.x = q['x']
        goal.pose.orientation.y = q['y']
        goal.pose.orientation.z = q['z']
        goal.pose.orientation.w = q['w']

        self._goal_pub.publish(goal)

        extent = obj.get('extent', {})
        self.get_logger().info(
            f'Navigating to "{label}" '
            f'at ({pos[0]:.2f}, {pos[1]:.2f}) '
            f'yaw={math.degrees(yaw):.1f}° '
            f'size={extent.get("width", 0):.2f}×{extent.get("depth", 0):.2f}×{extent.get("height", 0):.2f}m')


def main(args=None):
    rclpy.init(args=args)
    node = NavMissionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
