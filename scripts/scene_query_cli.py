#!/usr/bin/env python3
"""CLI for scene graph queries — zero ROS2 dependency."""

import argparse
import json
import sys
from scene_graph import SceneGraph


def parse_args():
    parser = argparse.ArgumentParser(description='Scene graph query tool')
    parser.add_argument('--json', required=True, help='Path to scene_graph.json')
    parser.add_argument('--query', required=True, help='Query string')
    parser.add_argument('--pretty', action='store_true', help='Pretty-print JSON')
    return parser.parse_args()


def execute_query(graph: SceneGraph, query_str: str) -> dict:
    """Parse query string and execute, returning {query, result, error?}."""
    parts = query_str.split()
    if not parts:
        return {'query': query_str, 'error': 'Empty query'}

    cmd = parts[0]

    try:
        if cmd == 'find':
            if len(parts) < 2:
                return {'query': query_str, 'error': 'Usage: find <label>'}
            label = parts[1]
            result = graph.find_by_label(label)
            return {'query': query_str, 'result': result}

        elif cmd == 'closest':
            if len(parts) < 2:
                return {'query': query_str, 'error': 'Usage: closest <label> [x y z]'}
            label = parts[1]
            reference = None
            if len(parts) >= 5:
                reference = [float(parts[2]), float(parts[3]), float(parts[4])]
            result = graph.closest(label, reference)
            return {'query': query_str, 'result': result}

        elif cmd == 'near':
            if len(parts) < 3:
                return {'query': query_str, 'error': 'Usage: near <label_a> <label_b>'}
            label_a, label_b = parts[1], parts[2]
            result = graph.find_near(label_a, label_b)
            return {'query': query_str, 'result': result}

        elif cmd == 'cluster':
            if len(parts) < 2:
                return {'query': query_str, 'error': 'Usage: cluster <radius>'}
            radius = float(parts[1])
            clusters = graph.cluster_by_proximity(radius)
            result = [{'cluster_id': i, 'size': len(c), 'nodes': c}
                      for i, c in enumerate(clusters)]
            return {'query': query_str, 'result': result}

        elif cmd == 'objects_near':
            if len(parts) < 3:
                return {'query': query_str, 'error': 'Usage: objects_near <label> <threshold>'}
            label = parts[1]
            threshold = float(parts[2])
            result = graph.objects_near(label, threshold)
            return {'query': query_str, 'result': result}

        elif cmd == 'is_between':
            if len(parts) < 4:
                return {'query': query_str, 'error': 'Usage: is_between <query_label> <label_a> <label_b> [width]'}
            query_label, label_a, label_b = parts[1], parts[2], parts[3]
            corridor_width = 0.5
            if len(parts) >= 5:
                corridor_width = float(parts[4])
            result = graph.is_between(query_label, label_a, label_b, corridor_width)
            return {'query': query_str, 'result': result}

        elif cmd == 'relative':
            if len(parts) < 3:
                return {'query': query_str, 'error': 'Usage: relative <node_id_a> <node_id_b>'}
            node_id_a, node_id_b = parts[1], parts[2]
            node_a = None
            node_b = None
            for node in graph._nodes:
                if node['id'] == node_id_a:
                    node_a = node
                if node['id'] == node_id_b:
                    node_b = node
            if not node_a or not node_b:
                return {'query': query_str, 'error': f'Node not found: {node_id_a or node_id_b}'}
            result = graph.relative_position(node_a, node_b)
            return {'query': query_str, 'result': result}

        else:
            return {'query': query_str, 'error': f'Unknown command: {cmd}'}

    except Exception as e:
        return {'query': query_str, 'error': f'{type(e).__name__}: {e}'}


def main():
    args = parse_args()

    try:
        graph = SceneGraph.from_json_file(args.json)
    except Exception as e:
        output = {'error': f'Failed to load graph: {e}'}
        print(json.dumps(output, indent=2 if args.pretty else None), file=sys.stdout)
        sys.exit(1)

    output = execute_query(graph, args.query)
    print(json.dumps(output, indent=2 if args.pretty else None), file=sys.stdout)

    # Exit with error code if query failed
    if 'error' in output:
        sys.exit(1)


if __name__ == '__main__':
    main()
