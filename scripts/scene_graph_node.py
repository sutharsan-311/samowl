#!/usr/bin/env python3
"""Scene graph node — Step 1: subscribe to /samowl/detections and validate."""

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class SceneGraphNode(Node):
    def __init__(self):
        super().__init__('scene_graph_node')
        self.sub_ = self.create_subscription(
            String,
            '/samowl/detections',
            self._on_detection,
            10)
        self.get_logger().info('Scene graph node ready, waiting for /samowl/detections')

    def _on_detection(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f'Malformed detection JSON: {exc}')
            return

        obj_id = data.get('id', '<unknown>')
        label = data.get('label', '<unknown>')
        score = data.get('score', 0.0)
        pos = data.get('position', [])
        self.get_logger().info(
            f'Detection: id={obj_id} label={label} '
            f'score={score:.2f} pos={pos}')


def main(args=None):
    rclpy.init(args=args)
    node = SceneGraphNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
