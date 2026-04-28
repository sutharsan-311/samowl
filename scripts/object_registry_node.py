#!/usr/bin/env python3
"""Object registry node — maintains a live map of detected objects with geometry.

Watches the hotspot JSON file written by samowl and re-publishes its contents
on /samowl/object_registry whenever a new detection arrives or the file changes.

Published topics
----------------
/samowl/object_registry  (std_msgs/String)
    JSON object: { "objects": [ <hotspot> ... ] }
    Each hotspot includes position_3d, extent, footprint_2d, and approach.

Subscribed topics
-----------------
/samowl/detections  (std_msgs/String)
    Triggers an immediate reload of the hotspot file on each detection.

Parameters
----------
hotspot_file  : str   — absolute path to hotspots.json (default: /tmp/samowl/hotspots.json)
poll_rate_hz  : float — background poll rate for file changes (default: 1.0)
"""

import json
import os
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class ObjectRegistryNode(Node):
    def __init__(self):
        super().__init__('object_registry_node')

        self.declare_parameter('hotspot_file', '/tmp/samowl/hotspots.json')
        self.declare_parameter('poll_rate_hz', 1.0)

        self._hotspot_file = self.get_parameter('hotspot_file').get_parameter_value().string_value
        poll_hz = self.get_parameter('poll_rate_hz').get_parameter_value().double_value

        self._pub = self.create_publisher(String, '/samowl/object_registry', 10)

        self._sub = self.create_subscription(
            String, '/samowl/detections', self._on_detection, 10)

        self._poll_timer = self.create_timer(1.0 / poll_hz, self._poll)

        self._last_mtime = 0.0
        self._last_registry: list = []

        self.get_logger().info(
            f'Object registry watching {self._hotspot_file}')

    # ------------------------------------------------------------------ #

    def _on_detection(self, _msg: String) -> None:
        self._reload()

    def _poll(self) -> None:
        try:
            mtime = os.path.getmtime(self._hotspot_file)
        except FileNotFoundError:
            return
        if mtime != self._last_mtime:
            self._reload()

    def _reload(self) -> None:
        try:
            data = json.loads(
                open(self._hotspot_file, encoding='utf-8').read())
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            self.get_logger().warn(f'Failed to read hotspot file: {exc}')
            return

        try:
            mtime = os.path.getmtime(self._hotspot_file)
        except FileNotFoundError:
            mtime = time.time()
        self._last_mtime = mtime

        objects = data.get('hotspots', [])
        self._last_registry = objects

        msg = String()
        msg.data = json.dumps({'objects': objects})
        self._pub.publish(msg)

        self.get_logger().info(
            f'Registry updated: {len(objects)} object(s)')


def main(args=None):
    rclpy.init(args=args)
    node = ObjectRegistryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
