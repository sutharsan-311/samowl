# Graph Report - .  (2026-04-23)

## Corpus Check
- 11 files · ~110,497 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 120 nodes · 177 edges · 8 communities detected
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 4 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## God Nodes (most connected - your core abstractions)
1. `main()` - 14 edges
2. `SceneGraphNode` - 11 edges
3. `SceneQueryNode` - 8 edges
4. `main()` - 8 edges
5. `main()` - 6 edges
6. `Predictor` - 6 edges
7. `SceneGraphNode (ROS2 Node)` - 6 edges
8. `NanoOwlPredictor` - 5 edges
9. `parse_args()` - 5 edges
10. `run_python()` - 5 edges

## Surprising Connections (you probably didn't know these)
- `CMake Installation (Scripts & Models)` --installs--> `Unix Socket Server (Daemon)`  [INFERRED]
  CMakeLists.txt → scripts/samowl_daemon.py
- `CMake Installation (Scripts & Models)` --installs--> `SceneGraphNode (ROS2 Node)`  [EXTRACTED]
  CMakeLists.txt → scripts/scene_graph_node.py
- `CMake Installation (Scripts & Models)` --installs--> `SceneQueryNode (ROS2 Node)`  [EXTRACTED]
  CMakeLists.txt → scripts/scene_query_node.py
- `CMake Build Target` --compiles--> `samowl main() — C++ ROS2 Executive`  [EXTRACTED]
  CMakeLists.txt → src/samowl.cpp
- `C++ ↔ Python Architecture (Hard Language Boundary)` --rationale_for--> `samowl main() — C++ ROS2 Executive`  [EXTRACTED]
  CLAUDE.md → src/samowl.cpp

## Hyperedges (group relationships)
- **Persistent Daemon Architecture** — samowl_daemon_bundle, samowl_daemon_socket, samowl_daemon_inference [EXTRACTED 0.95]
- **Scene Graph Detection + Visualization Pipeline** — scene_graph_node_class, scene_graph_node_sub, scene_graph_marker_pub, samowl_daemon_inference [INFERRED 0.85]
- **Scene Query Subsystem** — scene_query_node_class, scene_query_graph_sub, scene_query_find_near, scene_query_closest [EXTRACTED 0.90]

## Communities

### Community 0 - "C++ Executive"
Cohesion: 0.15
Nodes (21): bbox_to_points(), draw_boundary(), estimate_normal(), load_image_encoder_engine(), load_mask_decoder_engine(), main(), next_hotspot_id(), parse_args() (+13 more)

### Community 1 - "Persistent Daemon"
Cohesion: 0.1
Nodes (21): C++ ↔ Python Architecture (Hard Language Boundary), CMake Build Target, CMake Installation (Scripts & Models), samowl main() — C++ ROS2 Executive, Options Struct (Configuration), ModelBundle (Persistent Model Holder), YAML Config Loading (Daemon), Persistent Daemon Inference (+13 more)

### Community 2 - "Persistent Daemon"
Cohesion: 0.15
Nodes (17): get_param(), _handle_client(), _import_pipeline(), load_config(), main(), ModelBundle, parse_args(), Resolve a parameter with precedence: request JSON > YAML config > hardcoded defa (+9 more)

### Community 3 - "Build System"
Cohesion: 0.18
Nodes (14): build_request_json(), derive_camera_info_topic(), find_config(), find_script(), load_config(), main(), parse_args(), parse_response() (+6 more)

### Community 4 - "Scene Graph Subsystem"
Cohesion: 0.22
Nodes (7): _dist(), main(), Recompute all undirected "near" edges from scratch., Weighted blend: lean toward the new observation, keep some history.      alpha r, Return (node_id, distance) of the closest same-label node, or (None, inf)., SceneGraphNode, _update_position()

### Community 5 - "Community 5"
Cohesion: 0.23
Nodes (6): Node, main(), Return all nodes whose label matches, as clean output dicts., Return the node with matching label closest to reference (map frame), or None., Return pairs of node IDs where one has label_a, the other label_b, and a near ed, SceneQueryNode

### Community 6 - "ML Inference"
Cohesion: 0.4
Nodes (3): _load_nanoowl_encoder(), NanoOwlPredictor, OWL-ViT with TRT vision encoder and cached PyTorch text encoder.

### Community 7 - "Persistent Daemon"
Cohesion: 1.0
Nodes (2): Fork/Exec Per-Frame Bottleneck, Target: Persistent Python Daemon

## Knowledge Gaps
- **26 isolated node(s):** `Weighted blend: lean toward the new observation, keep some history.      alpha r`, `Return (node_id, distance) of the closest same-label node, or (None, inf).`, `Recompute all undirected "near" edges from scratch.`, `Return the node with matching label closest to reference (map frame), or None.`, `Return pairs of node IDs where one has label_a, the other label_b, and a near ed` (+21 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Persistent Daemon`** (2 nodes): `Fork/Exec Per-Frame Bottleneck`, `Target: Persistent Python Daemon`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Predictor` connect `C++ Executive` to `Persistent Daemon`?**
  _High betweenness centrality (0.093) - this node is a cross-community bridge._
- **Why does `ModelBundle (Persistent Model Holder)` connect `Persistent Daemon` to `C++ Executive`?**
  _High betweenness centrality (0.091) - this node is a cross-community bridge._
- **What connects `Weighted blend: lean toward the new observation, keep some history.      alpha r`, `Return (node_id, distance) of the closest same-label node, or (None, inf).`, `Recompute all undirected "near" edges from scratch.` to the rest of the system?**
  _26 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Persistent Daemon` be split into smaller, more focused modules?**
  _Cohesion score 0.1 - nodes in this community are weakly interconnected._