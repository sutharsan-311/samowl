#!/usr/bin/env python3
"""Scene query node — subscribe to /scene_graph and maintain query state."""

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from scene_graph import SceneGraph


class SceneQueryNode(Node):
    def __init__(self):
        super().__init__('scene_query_node')

        # Latest snapshot from /scene_graph — delegated to SceneGraph.
        self._graph = SceneGraph()

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

        nodes = data.get('nodes', [])
        edges = data.get('edges', [])
        self._graph.update(nodes, edges)

        self.get_logger().debug(f'Graph updated: {len(nodes)} nodes, {len(edges)} edges')

        # Debug: exercise find_by_label on every update.
        for label in {n['label'] for n in nodes}:
            results = self._graph.find_by_label(label)
            self.get_logger().debug(f'find_by_label({label!r}) → {results}')

        # Debug: exercise find_near for every distinct label pair in the graph.
        labels = list({n['label'] for n in nodes})
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                pairs = self._graph.find_near(labels[i], labels[j])
                if pairs:
                    self.get_logger().debug(
                        f'find_near({labels[i]!r}, {labels[j]!r}) → {pairs}')

        # Debug: closest node per label.
        for label in labels:
            result = self._graph.closest(label)
            if result:
                self.get_logger().debug(
                    f'closest({label!r}) → {result["id"]} ({result["distance"]:.2f} m)')



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
