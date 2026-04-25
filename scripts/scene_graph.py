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
