#!/usr/bin/env python3
"""Scene graph node — Step 6.2: state management and spatial edge construction."""

import json
import math
import time
from collections import defaultdict

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

MATCH_THRESHOLD = 0.5   # metres — same label within this radius → same node
NEAR_THRESHOLD  = 1.5   # metres — nodes within this radius get a "near" edge
NODE_TIMEOUT    = 3.0   # seconds — nodes unseen longer than this are pruned


def _dist(a, b) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _update_position(old_pos: list, new_pos: list) -> list:
    """Weighted blend: lean toward the new observation, keep some history.

    alpha rises to 0.9 for large jumps (> 1 m) so fast motion still tracks.
    """
    d = _dist(old_pos, new_pos)
    alpha = 0.9 if d > 1.0 else 0.7
    return [alpha * n + (1 - alpha) * o for o, n in zip(old_pos, new_pos)]


class SceneGraphNode(Node):
    def __init__(self):
        super().__init__('scene_graph_node')

        # { stable_id: {"id", "label", "position", "last_seen"} }
        self.nodes: dict = {}
        # [{"source", "target", "relation"}]
        self.edges: list = []

        # per-label counter to generate stable IDs (chair_0, chair_1, …)
        self._label_counters: dict = defaultdict(int)

        self._sub = self.create_subscription(
            String,
            '/samowl/detections',
            self._on_detection,
            10)

        self._pub = self.create_publisher(String, '/scene_graph', 10)

        self.get_logger().info('Scene graph node ready')

    # ------------------------------------------------------------------ #
    #  Subscription callback                                               #
    # ------------------------------------------------------------------ #

    def _on_detection(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f'Malformed detection JSON: {exc}')
            return

        label    = data.get('label', '')
        position = data.get('position', [])
        if not label or len(position) != 3:
            return

        self._upsert_node(label, position)
        self._cleanup_nodes()
        self._rebuild_edges()
        self._publish_graph()

        print(f'[GRAPH] nodes={len(self.nodes)} edges={len(self.edges)}')

    # ------------------------------------------------------------------ #
    #  State management                                                    #
    # ------------------------------------------------------------------ #

    def _upsert_node(self, label: str, position: list) -> None:
        """Find the nearest matching node or create a new one."""
        best_id   = None
        best_dist = float('inf')

        for node_id, node in self.nodes.items():
            if node['label'] != label:
                continue
            d = _dist(node['position'], position)
            if d < best_dist:
                best_dist = d
                best_id   = node_id

        now = time.time()

        if best_id is not None and best_dist < MATCH_THRESHOLD:
            node = self.nodes[best_id]
            node['position']  = _update_position(node['position'], position)
            node['last_seen'] = now
        else:
            new_id = f'{label}_{self._label_counters[label]}'
            self._label_counters[label] += 1
            self.nodes[new_id] = {
                'id':        new_id,
                'label':     label,
                'position':  list(position),
                'last_seen': now,
            }

    def _cleanup_nodes(self) -> None:
        now     = time.time()
        expired = [nid for nid, n in self.nodes.items()
                   if now - n['last_seen'] > NODE_TIMEOUT]
        for nid in expired:
            del self.nodes[nid]

    def _rebuild_edges(self) -> None:
        """Recompute all undirected "near" edges from scratch."""
        ids   = list(self.nodes.keys())
        edges = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a = self.nodes[ids[i]]
                b = self.nodes[ids[j]]
                if _dist(a['position'], b['position']) < NEAR_THRESHOLD:
                    edges.append({
                        'source':   ids[i],
                        'target':   ids[j],
                        'relation': 'near',
                    })
        self.edges = edges

    # ------------------------------------------------------------------ #
    #  Publishing                                                          #
    # ------------------------------------------------------------------ #

    def _publish_graph(self) -> None:
        payload = {
            'nodes': [
                {k: v for k, v in n.items() if k != 'last_seen'}
                for n in self.nodes.values()
            ],
            'edges': self.edges,
        }
        self._pub.publish(String(data=json.dumps(payload)))


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
