# 3DAeroVLM

**A 3D Vision-Language Benchmark for UAV-Based Post-Disaster Assessment**

The first benchmark combining 3D point cloud data with vision-language instruction pairs for disaster damage assessment. Built on [3DAeroRelief](https://github.com/BinaLab/3DAeroRelief) point clouds from Hurricane Ian (Florida, 2022) and following the task taxonomy of [DisasterM3](https://github.com/Junjue-Wang/DisasterM3) (NeurIPS 2025).

## Overview

| Property | Value |
|----------|-------|
| Point Cloud Scenes | 64 (8 areas) |
| Source Disaster | Hurricane Ian (2022), Florida |
| Semantic Classes | 5 (Building-Damage, Building-No-Damage, Road, Tree, Background) |
| Instruction Pairs | 1,330 |
| Train Set | 1,174 pairs (56 scenes, Areas 1, 3-8) |
| Test Set | 156 pairs (8 scenes, Area 2) |
| Task Types | 5 |
| Input Modality | 3D point cloud (x, y, z, r, g, b, label) |

## Task Types

Following DisasterM3's taxonomy, adapted for 3D:

| Task | Count | Format | Example |
|------|-------|--------|---------|
| Scene Recognition | 256 | Open-ended + MC | "What structures are present in this 3D scene?" |
| Damage Counting | 381 | Multiple-choice | "How many damaged buildings are in this scene?" |
| Referring Segmentation | 455 | Segmentation | "Segment the damaged buildings in this scene." |
| Spatial Reasoning | 110 | Open-ended | "What is the distance between the nearest damaged and intact building?" |
| Damage Report | 128 | Open-ended | "Generate a comprehensive damage assessment report." |

## Dataset Files

```
data/
├── scene_facts.json           # Per-scene ground truth facts (building counts, distances, etc.)
├── 3daero_vlm_instruct.json   # Training instruction pairs (Areas 1, 3-8)
├── 3daero_vlm_bench.json      # Test instruction pairs (Area 2)
└── 3daero_vlm_all.json        # Combined (all 64 scenes)
```

### Instruction Pair Format

```json
{
  "scene_id": "Area_1/pp1.ply",
  "task_type": "counting",
  "question": "How many damaged buildings are in this 3D scene?",
  "options": ["A. 3", "B. 6", "C. 4", "D. 2", "E. 5"],
  "answer": "C",
  "ground_truth_value": 4,
  "split": "train",
  "modality": "3D_point_cloud",
  "format": "multiple_choice"
}
```

## Ground Truth Generation

Building counts and spatial facts are computed automatically from 3DAeroRelief's per-point semantic labels:

1. **Class Distribution**: Point counts per semantic class (direct from labels)
2. **Building Counting**: 3D DBSCAN clustering (eps=0.5) on labeled building points, with adaptive minimum cluster size (0.3% of subsampled points, clamped to [30, 150])
3. **Spatial Relationships**: Euclidean distances between building cluster centroids and road points
4. **Multiple-Choice Options**: Following DisasterM3's approach with controlled deviations (+-20%, +-40%)

## Project Structure

```
3DAeroVLM/
├── README.md
├── PROJECT_GUIDE.md                # Detailed project guide with next steps
├── ground_truth/
│   └── extract_facts.py            # Extracts building counts, distances from labeled PLY files
├── instruction_gen/
│   └── generate_pairs.py           # Generates QA pairs from scene facts
├── evaluation/                     # VLM benchmarking (TODO)
├── data/                           # Generated datasets
└── visualizations/                 # Cluster verification images
```

## Quick Start

```bash
# Activate environment
source /media/volume/Tene_Volume/venv/bin/activate
cd /media/volume/Tene_Volume/3DAeroVLM

# Step 1: Extract ground truth facts from point clouds
python ground_truth/extract_facts.py \
    --input ../processed_data \
    --output data/scene_facts.json

# Step 2: Generate instruction pairs
python instruction_gen/generate_pairs.py \
    --facts data/scene_facts.json \
    --output-train data/3daero_vlm_instruct.json \
    --output-test data/3daero_vlm_bench.json \
    --output-all data/3daero_vlm_all.json
```

## Data Dependencies

This project requires the processed 3DAeroRelief dataset:

```
/media/volume/Tene_Volume/
├── 3DAeroRelief_Dataset/    # Raw PLY files (downloaded from Dropbox)
└── processed_data/          # Merged labeled PLY files (from add_labels_and_viz.py)
    ├── Area_1/ ... Area_8/  # 64 scenes total
```

Each processed PLY file contains per-point fields: `(x, y, z, r, g, b, label)` where label is:
- 0: Background
- 1: Building-Damage
- 2: Building-No-Damage
- 3: Road
- 4: Tree

## Novelty

No existing work combines 3D point clouds with vision-language tasks for disaster assessment:

- **3DAeroRelief** provides 3D point clouds but has no language component
- **DisasterM3** provides VLM instruction pairs but only for 2D satellite images
- Indoor 3D VLM benchmarks (ScanQA, SQA3D) don't address disaster scenarios

3DAeroVLM bridges this gap, enabling VLMs to reason about 3D disaster scenes through natural language with capabilities unique to 3D: spatial reasoning with true distances, building clustering from point geometry, and multi-view scene understanding.

## References

- **3DAeroRelief**: Le, N., Karimi, E., Rahnemoonfar, M. (2025). "3DAeroRelief: A Large-Scale 3D Semantic Segmentation Benchmark for Post-Disaster Assessment." [GitHub](https://github.com/BinaLab/3DAeroRelief)
- **DisasterM3**: Wang, J. et al. (NeurIPS 2025). "DisasterM3: A Remote Sensing Vision-Language Dataset for Disaster Damage Assessment and Response." [GitHub](https://github.com/Junjue-Wang/DisasterM3)

## Authors

Raphael — Bina Labs, Lehigh University
Advisor: Prof. Maryam Rahnemoonfar
