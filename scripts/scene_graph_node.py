#!/usr/bin/env python3
"""Scene graph node — Step 6.3: RViz MarkerArray visualization."""

import json
import math
import os
import time
from collections import defaultdict

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

# Hysteresis pair: existing nodes use a looser radius (harder to lose), new
# nodes require a stricter radius (harder to create a duplicate).
MATCH_THRESHOLD  = 0.6  # metres — update existing node if closest is within this
CREATE_THRESHOLD = 0.5  # metres — create a new node only if no existing within this
NEAR_THRESHOLD   = 1.5  # metres — nodes within this radius get a "near" edge
NODE_TIMEOUT     = 3.0  # seconds — nodes unseen longer than this are pruned
MAX_SPEED        = 5.0  # m/s   — clamp; faster observations are sensor noise

_LABEL_PRIORITY: dict = {"hospital bed": 10, "stretcher": 8, "chair": 5, "table": 5}


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

        # Load thresholds from YAML config; fall back to module-level defaults.
        self._near_threshold   = NEAR_THRESHOLD
        self._match_threshold  = MATCH_THRESHOLD
        self._create_threshold = CREATE_THRESHOLD
        try:
            import yaml
            from ament_index_python.packages import get_package_share_directory
            _cfg_path = os.path.join(get_package_share_directory('samowl'), 'config', 'samowl.yaml')
            with open(_cfg_path) as _f:
                _cfg = yaml.safe_load(_f)
            if _cfg and 'graph' in _cfg:
                _g = _cfg['graph']
                self._near_threshold   = float(_g.get('near_threshold',   self._near_threshold))
                self._match_threshold  = float(_g.get('match_threshold',  self._match_threshold))
                self._create_threshold = float(_g.get('create_threshold', self._create_threshold))
            self.get_logger().info(
                f'Graph thresholds: near={self._near_threshold} match={self._match_threshold} '
                f'create={self._create_threshold}')
        except Exception as exc:
            self.get_logger().warn(f'Could not load graph config: {exc} — using defaults')

        # { stable_id: {"id", "label", "position", "velocity", "last_seen"} }
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

        self._pub         = self.create_publisher(String, '/scene_graph', 10)
        self._marker_pub  = self.create_publisher(
            MarkerArray, '/scene_graph_markers', 10)

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
        if len(position) == 3 and max(abs(v) for v in position) < 1e-3:
            self.get_logger().warn(f'Rejecting origin detection for label={label!r}')
            return
        if not label or len(position) != 3:
            return

        self._upsert_node(label, position)
        self._cleanup_nodes()
        self._rebuild_edges()
        self._publish_graph()
        self._publish_markers()

        print(f'[GRAPH] nodes={len(self.nodes)} edges={len(self.edges)}')

    # ------------------------------------------------------------------ #
    #  State management                                                    #
    # ------------------------------------------------------------------ #

    def _find_best_match(self, label: str, position: list):
        """Return (node_id, distance) of the closest node within create_threshold.

        Same-label nodes take priority; cross-label nodes are also returned so
        that _upsert_node can relabel or suppress duplicates at the same position.
        """
        best_id       = None
        best_dist     = float('inf')
        same_label_id = None
        same_label_d  = float('inf')
        for node_id, node in self.nodes.items():
            d = _dist(node['position'], position)
            if node['label'] == label:
                if d < same_label_d:
                    same_label_d  = d
                    same_label_id = node_id
            if d < best_dist:
                best_dist = d
                best_id   = node_id
        # Prefer a same-label node when it is a viable match.
        if same_label_id is not None and same_label_d < self._match_threshold:
            return same_label_id, same_label_d
        return best_id, best_dist

    def _upsert_node(self, label: str, position: list) -> None:
        best_id, best_dist = self._find_best_match(label, position)
        now = time.time()

        # Hysteresis: match if an existing node is within MATCH_THRESHOLD (0.6 m).
        # Only create a new node if the closest candidate is beyond CREATE_THRESHOLD
        # (0.5 m) — the gap prevents oscillation at the boundary.
        if best_id is not None and best_dist < self._match_threshold:
            node = self.nodes[best_id]
            dt   = now - node['last_seen']

            if dt > 0:
                raw_vel = [(p - q) / dt
                           for p, q in zip(position, node['position'])]
                speed = math.sqrt(sum(v ** 2 for v in raw_vel))
                if speed <= MAX_SPEED:
                    old_vel = node['velocity']
                    node['velocity'] = [0.7 * nv + 0.3 * ov
                                        for nv, ov in zip(raw_vel, old_vel)]
                    print(f'[GRAPH] Matched {label} → {best_id} '
                          f'({best_dist:.2f} m) speed={speed:.2f} m/s')

            # Cross-label upgrade: relabel the existing node if the incoming
            # detection has a higher-priority label (e.g. "hospital bed" > "chair").
            if node['label'] != label and \
                    _LABEL_PRIORITY.get(label, 0) > _LABEL_PRIORITY.get(node['label'], 0):
                print(f'[GRAPH] Relabeled {best_id}: {node["label"]!r} → {label!r}')
                node['label'] = label

            node['position']  = _update_position(node['position'], position)
            node['last_seen'] = now
        elif best_dist >= self._create_threshold:
            new_id = f'{label}_{self._label_counters[label]}'
            self._label_counters[label] += 1
            self.nodes[new_id] = {
                'id':        new_id,
                'label':     label,
                'position':  list(position),
                'velocity':  [0.0, 0.0, 0.0],
                'last_seen': now,
            }
            print(f'[GRAPH] Created new node → {new_id}')

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
                if _dist(a['position'], b['position']) < self._near_threshold:
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

    def _publish_markers(self) -> None:
        now = self.get_clock().now().to_msg()
        array = MarkerArray()

        # Clear all previous markers atomically.
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        array.markers.append(delete_all)

        node_list = list(self.nodes.values())
        node_index = {n['id']: i for i, n in enumerate(node_list)}

        for i, node in enumerate(node_list):
            px, py, pz = node['position']

            # Blue sphere at the object centroid.
            sphere = Marker()
            sphere.header.frame_id = 'map'
            sphere.header.stamp    = now
            sphere.ns              = 'sg_nodes'
            sphere.id              = i
            sphere.type            = Marker.SPHERE
            sphere.action          = Marker.ADD
            sphere.pose.position.x = px
            sphere.pose.position.y = py
            sphere.pose.position.z = pz
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.2
            sphere.color.r = 0.2
            sphere.color.g = 0.4
            sphere.color.b = 1.0
            sphere.color.a = 0.85
            array.markers.append(sphere)

            # Label text floating above the sphere.
            label = Marker()
            label.header.frame_id = 'map'
            label.header.stamp    = now
            label.ns              = 'sg_labels'
            label.id              = i
            label.type            = Marker.TEXT_VIEW_FACING
            label.action          = Marker.ADD
            label.pose.position.x = px
            label.pose.position.y = py
            label.pose.position.z = pz + 0.3
            label.pose.orientation.w = 1.0
            label.scale.z         = 0.12
            label.color.r = label.color.g = label.color.b = 1.0
            label.color.a = 1.0
            label.text            = node['id']
            array.markers.append(label)

        # Yellow LINE_LIST connecting "near" node pairs.
        if self.edges:
            lines = Marker()
            lines.header.frame_id = 'map'
            lines.header.stamp    = now
            lines.ns              = 'sg_edges'
            lines.id              = 0
            lines.type            = Marker.LINE_LIST
            lines.action          = Marker.ADD
            lines.scale.x         = 0.03
            lines.color.r         = 1.0
            lines.color.g         = 0.85
            lines.color.b         = 0.0
            lines.color.a         = 0.8
            lines.pose.orientation.w = 1.0

            for edge in self.edges:
                src = self.nodes[edge['source']]['position']
                tgt = self.nodes[edge['target']]['position']
                p0, p1 = Point(), Point()
                p0.x, p0.y, p0.z = src
                p1.x, p1.y, p1.z = tgt
                lines.points.extend([p0, p1])

            array.markers.append(lines)

        self._marker_pub.publish(array)


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
