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

        # Debug: exercise find_by_label on every update.
        for label in {n['label'] for n in self._nodes}:
            results = self.find_by_label(label)
            self.get_logger().debug(f'find_by_label({label!r}) → {results}')

        # Debug: exercise find_near for every distinct label pair in the graph.
        labels = list({n['label'] for n in self._nodes})
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                pairs = self.find_near(labels[i], labels[j])
                if pairs:
                    self.get_logger().debug(
                        f'find_near({labels[i]!r}, {labels[j]!r}) → {pairs}')

    # ------------------------------------------------------------------ #
    #  Queries                                                             #
    # ------------------------------------------------------------------ #

    def find_near(self, label_a: str, label_b: str) -> list:
        """Return pairs of node IDs where one has label_a, the other label_b, and a near edge exists."""
        id_to_node = {n['id']: n for n in self._nodes}
        results    = []
        seen       = set()

        for edge in self._edges:
            if edge.get('relation') != 'near':
                continue

            na = id_to_node.get(edge.get('source'))
            nb = id_to_node.get(edge.get('target'))
            if not na or not nb:
                continue

            la, lb = na.get('label'), nb.get('label')
            if (la == label_a and lb == label_b) or (la == label_b and lb == label_a):
                key = tuple(sorted([na['id'], nb['id']]))
                if key not in seen:
                    seen.add(key)
                    results.append({'a': na['id'], 'b': nb['id']})

        return results

    def find_by_label(self, label: str) -> list:
        """Return all nodes whose label matches, as clean output dicts."""
        return [
            {
                'id':       node['id'],
                'position': node['position'],
                'velocity': node.get('velocity', [0.0, 0.0, 0.0]),
            }
            for node in self._nodes
            if node['label'] == label
        ]


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
