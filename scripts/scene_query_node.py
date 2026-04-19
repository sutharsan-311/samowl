#!/usr/bin/env python3
"""Scene query node — Step 6.5 Step 1: subscribe to /scene_graph and maintain local state."""

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class SceneQueryNode(Node):
    def __init__(self):
        super().__init__('scene_query_node')

        # Latest snapshot from /scene_graph — read-only view of graph state.
        self._nodes: list = []  # [{"id", "label", "position", "velocity"}]
        self._edges: list = []  # [{"source", "target", "relation"}]

        self._sub = self.create_subscription(
            String,
            '/scene_graph',
            self._on_graph,
            10)

        self.get_logger().info('Scene query node ready, listening on /scene_graph')

    # ------------------------------------------------------------------ #
    #  Graph subscription                                                  #
    # ------------------------------------------------------------------ #

    def _on_graph(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f'Malformed graph JSON: {exc}')
            return

        self._nodes = data.get('nodes', [])
        self._edges = data.get('edges', [])

        self.get_logger().debug(
            f'Graph updated: {len(self._nodes)} nodes, {len(self._edges)} edges')


def main(args=None):
    rclpy.init(args=args)
    node = SceneQueryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
