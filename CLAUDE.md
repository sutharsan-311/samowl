# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## 📦 Project Context

### Build

The workspace root is one level up from this package (`/home/susan/nano`). Always build from there:

```bash
cd /home/susan/nano
colcon build --packages-select samowl
source install/setup.bash
```

Run linting (the only test target currently configured):

```bash
colcon test --packages-select samowl
colcon test-result --verbose
```

---

## 🏗️ Architecture

samowl is two cooperating processes with a hard language boundary:

**C++ executable (`src/samowl.cpp`)** — owns everything ROS2: topic subscription, TF lookup, image serialization, process lifecycle. It never touches ML. It communicates with Python via `fork/execvp` and the filesystem.

**Python bridge (`scripts/samowl_pipeline.py`)** — owns all ML inference:

* OWL-ViT detection (Transformers)
* SAM segmentation (TensorRT via torch2trt)
* 3D projection and hotspot fusion

This is a standalone CLI script with no ROS2 dependency.

---

### 🔗 How they connect

* `run_python()` builds CLI args → `fork()` → `execvp()`
* Python returns only exit code (metadata is currently discarded in topic mode)
* Communication is file-based (`/tmp/samowl`)

---

### ⚙️ Operating Modes

**File mode (`--image`)**

* Single run → exits

**Topic mode (`--rgb-topic`, `--depth-topic`)**

* Subscribes via ApproximateTime sync
* Saves frame → `/tmp/samowl`
* Looks up TF transform
* Calls Python pipeline
* Drops frames during processing

---

### 📂 Model Files

Located in `data/` → installed to `share/samowl/data/`:

* `owlvit-base-patch32/` (HuggingFace)
* `resnet18_image_encoder.engine` (TensorRT)
* `mobile_sam_mask_decoder.engine` (TensorRT)

⚠️ TensorRT engines:

* Not generated in this repo
* Must match hardware + TensorRT version exactly

---

### 🔄 Data Flow (per frame)

```
RGB + Depth → save PNG → write camera JSON
→ fork Python → OWL-ViT detect → SAM segment
→ 3D projection → hotspot JSON
```

---

### ⚠️ Known Constraints

* Python process restarts every frame (major latency)
* `/tmp/samowl` is never cleaned
* Only best detection is used
* `depth_scale` is hardcoded to 0.001
* `geometry_msgs` is transitively included (not explicit)

---

### 🧠 Graphify Note

This architecture must always be analyzed using Graphify outputs to understand:

* real dependencies
* hidden coupling
* bottlenecks

---

## 🧠 AI Operating Rules (IMPORTANT)

These rules must be applied when working with the architecture and constraints described above.

---

### 🔍 Mandatory Workflow

#### 1. ALWAYS USE GRAPHIFY FIRST

Before answering ANY architecture or code question:

* Read:

  * `/graphify-out/graph.html`
  * `/graphify-out/GRAPH_REPORT.md`
  * `/graphify-out/graph.json`

Use Graphify to:

* detect “god nodes”
* trace dependencies
* understand cross-module relationships

Do NOT rely only on local file reading.

---

#### 2. THINK IN PIPELINES

Current:
C++ → Python → OWL-ViT → SAM

Target:
Sensors → Detection → Segmentation → Fusion → Scene Graph → Consumers

---

#### 3. ALWAYS CHECK RISKS

For every change:

* Hidden dependencies (models, tokenizer, TRT engines)
* Runtime bottlenecks (model loading, IPC, disk I/O)
* Coupling (e.g., `main()`)
* Fragile data flow

---

## 🚨 Critical Issues (DO NOT IGNORE)

* Python launched per frame via `fork/execvp` → major bottleneck
* `main()` acts as a god node
* stdout metadata is lost
* TensorRT engines not reproducible
* `/tmp/samowl` grows indefinitely

🚨 CRITICAL:
The fork/exec design is the **primary system bottleneck** and must not be extended.

---

## 🏗️ Target Architecture

Refactor toward modular ROS2 system:

* detector node (OWL-ViT)
* segmenter node (SAM)
* fusion node (3D + filtering)
* scene graph node (relationships)

Avoid monolithic pipelines.

---

## 🔗 Data Contracts

Use structured outputs:

```
{
  "id": "object_1",
  "label": "chair",
  "position": [x, y, z],
  "confidence": 0.9
}
```

---

## 🚫 Anti-Patterns

* Expanding `main()`
* Per-frame process spawning
* Hardcoded model paths
* Silent failures
* File-based IPC as primary system

---

## ⚙️ Engineering Rules

* Use persistent processes
* Prefer IPC over disk
* Make dependencies explicit
* Validate all inputs

---

## 🧪 When Modifying Code

Always:

1. Use Graphify to trace impact
2. Check dependencies
3. Avoid increasing coupling
4. Explain impact on:

   * performance
   * scalability
   * maintainability

---

## 🔄 Graphify Update Rule

At end of each session:

1. Regenerate Graphify outputs
2. Update `/graphify-out/`
3. Review:

   * node centrality
   * dependency changes

Explain architecture impact.

---

## 🚀 Priority Roadmap

1. Replace fork/exec with persistent Python process
2. Add JSON IPC (bidirectional)
3. Publish ROS2 topics
4. Remove file-based pipeline
5. Modularize system

---

## 💬 Interaction Style

* Be direct and critical
* Avoid generic advice
* Reference real components
* Suggest concrete changes

---

### 🤖 PARALLEL AGENT EXECUTION (HIGH PRIORITY)

When solving complex tasks (refactoring, debugging, architecture changes), you MUST decompose the work into multiple parallel agents.

#### When to use parallel agents

Use parallel agents if the task involves:

* Multiple files or modules
* Cross-language boundaries (C++ ↔ Python)
* Performance + architecture + dependency analysis
* Refactoring pipelines or system design

---

#### Required Agent Roles

Split work into independent agents such as:

1. **Architecture Agent**

   * Analyzes Graphify output
   * Identifies coupling, god nodes, and system structure

2. **Performance Agent**

   * Focuses on bottlenecks (e.g., fork/exec, model loading, I/O)
   * Suggests latency and throughput improvements

3. **Dependency Agent**

   * Finds hidden dependencies (TensorRT, tokenizer, models)
   * Checks reproducibility and environment risks

4. **Implementation Agent**

   * Proposes concrete code changes
   * Ensures alignment with architecture goals

---

#### Execution Rules

* Each agent must:

  * Work independently
  * Produce its own findings
  * Reference real components and Graphify insights

* After all agents finish:

  * Merge outputs into a single coherent plan
  * Resolve conflicts between agents
  * Prioritize actions based on impact

---

#### Output Format

When using parallel agents, structure responses as:

Architecture Agent:
...

Performance Agent:
...

Dependency Agent:
...

Implementation Agent:
...

Final Plan:
...

---

#### Constraints

* Do NOT skip Graphify analysis
* Do NOT produce generic answers
* Do NOT merge everything into one vague explanation
* Each agent must add unique value

---

#### Goal

Simulate a **senior engineering team working in parallel**, not a single linear thinker.

This is required for all major system-level tasks.

## 🧾 Version Control Rules (MANDATORY)

All changes must be tracked with Git in a clean, structured manner.

---

### 🔄 Commit After Each Logical Fix

After completing any meaningful change (bug fix, refactor step, feature addition):

1. Stage relevant files only (avoid unrelated changes)
2. Create a commit with a clear, descriptive message
3. Ensure the repository remains in a working state

---

### 🧠 Commit Strategy

Each commit must represent:

* One logical change
* One clear purpose

Do NOT:

* bundle multiple unrelated fixes
* commit broken or partial implementations

---

### 📝 Commit Message Format

Use structured messages:

```
<type>: <short summary>

Details:
- What was changed
- Why it was changed
- Impact (performance / architecture / fix)
```

---

### 📌 Commit Types

* `fix:` bug fixes
* `refactor:` structural improvements
* `perf:` performance improvements
* `feat:` new features
* `chore:` setup, configs, cleanup

---

### 🔍 Example

```
perf: replace fork/exec with persistent Python process

Details:
- Added Python daemon server
- Replaced per-frame process spawning
- Reduced inference latency significantly
```

---

### 🔄 Workflow Integration

For every session:

1. Make changes
2. Validate changes
3. Update Graphify outputs
4. Commit changes

---

### 🚨 Constraints

* Do NOT skip commits
* Do NOT commit generated artifacts (unless required)
* Do NOT commit large model files (use external storage or Git LFS)

---

### 🎯 Goal

Maintain a clean, traceable history that reflects architectural evolution step-by-step.

### 🚫 No Co-Author Attribution (STRICT)

Do NOT include any of the following in commit messages:

* "Co-authored-by"
* "Generated by Claude"
* Any AI attribution or signature

All commits must appear as if authored solely by the developer.

---

### 📌 Rules

* Commit messages must contain only:

  * type
  * summary
  * details (if needed)

* No extra metadata, signatures, or attribution lines

---

### 🎯 Goal

Maintain a clean, professional git history without AI attribution noise.

### 🔄 Incremental Commit Enforcement (STRICT)

All fixes MUST be committed incrementally.

---

#### Rules

* Each fix = ONE commit
* Do NOT combine multiple fixes into a single commit
* Do NOT produce a “final consolidated commit”

---

#### Required Behavior

When solving multi-step problems:

1. Break solution into ordered steps
2. After EACH step:

   * Apply code changes
   * Provide commit message
   * Stop and wait before next step (unless explicitly told to continue)

---

#### Example (CORRECT)

Step 1: Replace fork/exec
→ commit

Step 2: Add Python daemon
→ commit

Step 3: Add JSON IPC
→ commit

---

#### Example (WRONG)

❌ One commit containing:

* daemon
* IPC
* cleanup
* dependency fixes

---

#### Output Format Requirement

For every step:

Step X:

* Changes:
* Files modified:
* Why:

Commit message:

```
type: summary

Details:
...
```

---

#### Goal

Create a clean, step-by-step evolution of the system where each commit is:

* isolated
* reversible
* understandable
