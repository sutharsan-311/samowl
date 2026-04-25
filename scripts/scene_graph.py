#!/usr/bin/env python3
"""Pure Python scene graph — query logic independent of ROS2."""

import json
import math
from typing import List, Dict, Optional, Tuple


class SceneGraph:
    """Immutable snapshot of scene graph state with spatial query methods."""

    def __init__(self, nodes: Optional[List[Dict]] = None, edges: Optional[List[Dict]] = None):
        """Initialize with nodes and edges lists."""
        self._nodes = nodes or []
        self._edges = edges or []

    @classmethod
    def from_dict(cls, data: Dict) -> "SceneGraph":
        """Construct from dict (typically JSON-decoded message)."""
        return cls(
            nodes=data.get('nodes', []),
            edges=data.get('edges', [])
        )

    @classmethod
    def from_json_file(cls, path: str) -> "SceneGraph":
        """Load scene graph from JSON file (scan mode output or exported state)."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def update(self, nodes: List[Dict], edges: List[Dict]) -> None:
        """Replace graph state with new nodes and edges."""
        self._nodes = nodes
        self._edges = edges

    @staticmethod
    def _dist(a: List[float], b: List[float]) -> float:
        """Euclidean distance in 3D."""
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    @staticmethod
    def _dist_2d(a: List[float], b: List[float]) -> float:
        """Euclidean distance in XY plane (Z ignored)."""
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def find_by_label(self, label: str) -> List[Dict]:
        """Return all nodes with matching label as clean dicts {id, position, velocity}."""
        return [
            {
                'id': node['id'],
                'position': node['position'],
                'velocity': node.get('velocity', [0.0, 0.0, 0.0]),
            }
            for node in self._nodes
            if node['label'] == label
        ]

    def find_near(self, label_a: str, label_b: str) -> List[Dict]:
        """Return pairs of node IDs where one has label_a, other has label_b, and near edge exists."""
        id_to_node = {n['id']: n for n in self._nodes}
        results = []
        seen = set()

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

    def closest(self, label: str, reference: Optional[List[float]] = None) -> Optional[Dict]:
        """Return node with matching label closest to reference point (map frame), or None."""
        if reference is None:
            reference = [0.0, 0.0, 0.0]

        candidates = [n for n in self._nodes if n.get('label') == label]
        if not candidates:
            return None

        best = min(candidates, key=lambda n: self._dist(n['position'], reference))
        return {
            'id': best['id'],
            'position': best['position'],
            'distance': self._dist(best['position'], reference),
        }

    def cluster_by_proximity(self, radius: float) -> List[List[Dict]]:
        """Group nodes into clusters using Union-Find on distance threshold.

        Returns list of clusters ordered by size (largest first).
        Each cluster is a list of node dicts {id, label, position, ...}.
        """
        if not self._nodes:
            return []

        # Union-Find data structure
        parent = list(range(len(self._nodes)))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Union any pair of nodes within radius
        for i in range(len(self._nodes)):
            for j in range(i + 1, len(self._nodes)):
                dist = self._dist(self._nodes[i]['position'], self._nodes[j]['position'])
                if dist <= radius:
                    union(i, j)

        # Group nodes by root parent
        clusters_dict = {}
        for i, node in enumerate(self._nodes):
            root = find(i)
            if root not in clusters_dict:
                clusters_dict[root] = []
            clusters_dict[root].append(node)

        # Sort clusters by size (largest first)
        clusters = sorted(clusters_dict.values(), key=len, reverse=True)
        return clusters

    def objects_near(self, label: str, threshold: float) -> List[Dict]:
        """Find all nodes within threshold distance of any node with matching label.

        Returns list of node dicts (deduplicated by id).
        """
        anchors = [n for n in self._nodes if n['label'] == label]
        if not anchors:
            return []

        nearby = {}  # id -> node dict
        for anchor in anchors:
            for node in self._nodes:
                if node['id'] != anchor['id']:  # exclude anchor itself
                    dist = self._dist(anchor['position'], node['position'])
                    if dist <= threshold:
                        nearby[node['id']] = node

        return list(nearby.values())

    def is_between(self, query_label: str, label_a: str, label_b: str,
                   corridor_width: float = 0.5) -> List[Dict]:
        """Find query nodes in 2D corridor between label_a and label_b nodes.

        2D projection (Z ignored). Returns deduplicated list of query nodes.
        For each (anchor_a, anchor_b) pair, find query nodes within corridor_width.
        """
        nodes_a = [n for n in self._nodes if n['label'] == label_a]
        nodes_b = [n for n in self._nodes if n['label'] == label_b]
        query_nodes = [n for n in self._nodes if n['label'] == query_label]

        if not nodes_a or not nodes_b or not query_nodes:
            return []

        results = {}  # id -> node dict

        for anchor_a in nodes_a:
            for anchor_b in nodes_b:
                # 2D positions (x, y only)
                pa = anchor_a['position'][:2]
                pb = anchor_b['position'][:2]
                ab_len_sq = (pb[0] - pa[0]) ** 2 + (pb[1] - pa[1]) ** 2

                if ab_len_sq < 1e-6:  # degenerate case: a and b coincide
                    continue

                for query in query_nodes:
                    pq = query['position'][:2]

                    # Vector from A to Q
                    aq = [pq[0] - pa[0], pq[1] - pa[1]]
                    # Vector from A to B
                    ab = [pb[0] - pa[0], pb[1] - pa[1]]

                    # Scalar projection: t = dot(AQ, AB) / |AB|^2
                    t = (aq[0] * ab[0] + aq[1] * ab[1]) / ab_len_sq

                    # Check if t is in [0, 1] (between A and B)
                    if not (0.0 <= t <= 1.0):
                        continue

                    # Closest point on segment AB
                    closest = [pa[0] + t * ab[0], pa[1] + t * ab[1]]

                    # Perpendicular distance from Q to line AB
                    perp_dist = math.sqrt((pq[0] - closest[0]) ** 2 + (pq[1] - closest[1]) ** 2)

                    if perp_dist <= corridor_width:
                        results[query['id']] = query

        return list(results.values())
