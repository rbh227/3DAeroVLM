# 3DAeroVLM — Project Guide

**First 3D Vision-Language Benchmark for UAV Disaster Assessment**

Raphael — Bina Labs, Prof. Rahnemoonfar
Started: March 2026

---

## What This Project Is

We are building a **benchmark dataset** of question-answer pairs tied to 3D point cloud scenes from Hurricane Ian. The point clouds come from Nhut Le's 3DAeroRelief dataset (already labeled). Our contribution is adding the **vision-language layer** on top — following DisasterM3's task taxonomy but adapted for 3D.

```
Nhut's work:    UAV drone video  -->  COLMAP  -->  64 labeled 3D point clouds
Our work:       64 labeled point clouds  -->  instruction pairs + VLM benchmarking
```

We do NOT need to run COLMAP or re-annotate. The labels are done.

---

## Directory Structure

```
/media/volume/Tene_Volume/
├── venv/                          # Python virtual environment (always activate first)
├── 3DAeroRelief/                  # Nhut's cloned repo (preprocessing script)
├── 3DAeroRelief_Dataset/          # Raw data (pp*.ply + segmentpp*.ply), 8 areas
├── processed_data/                # Merged labeled point clouds (output of add_labels_and_viz.py)
│   ├── Area_1/ ... Area_8/       # Each has pp*.ply, pp*_vis_labels.ply, pp*_vis_rgb.ply
├── visualizations/                # Rendered PNG views of point clouds
├── 3DAeroVLM/                     # ** OUR PROJECT **
│   ├── PROJECT_GUIDE.md           # This file
│   ├── ground_truth/
│   │   └── extract_facts.py       # Extracts building counts, heights, spatial info from labels
│   ├── data/
│   │   └── scene_facts.json       # Extracted facts for all 64 scenes (generated)
│   ├── instruction_gen/           # Question-answer pair generation scripts
│   ├── evaluation/                # VLM benchmarking scripts
│   └── visualizations/            # Cluster verification images
```

## How to Start Working

```bash
# Always activate the virtual environment first
source /media/volume/Tene_Volume/venv/bin/activate
cd /media/volume/Tene_Volume/3DAeroVLM
```

---

## What Has Been Done

### 1. Dataset Downloaded and Processed
- Downloaded all 8 areas from Dropbox (~20GB)
- Ran Nhut's `add_labels_and_viz.py` to merge geometry + labels into single PLY files
- Result: 64 processed point clouds in `processed_data/`, each with (x, y, z, r, g, b, label)
- Labels: 0=Background, 1=Building-Damage, 2=Building-No-Damage, 3=Road, 4=Tree

### 2. Ground Truth Fact Extraction (extract_facts.py)
- Extracts per-scene facts from the labeled point clouds
- Uses **3D DBSCAN clustering** (eps=0.5, min_samples=20) to count individual buildings
  - Clusters in 3D (x,y,z) not just 2D — height differences help separate buildings
  - Subsamples to 50K points for memory safety
  - Filters out clusters < 200 points to remove debris fragments
- For each scene computes:
  - Building counts (damaged, intact, total)
  - Building heights, volumes, bounding boxes, centroids
  - Damage ratio
  - Spatial relationships (distances between damaged/intact, distance to road)
  - Class distributions (% of each label)
  - Road and tree presence
- Output: `data/scene_facts.json` — 64 entries, one per scene

**Current stats from extraction:**
- 64 scenes processed
- 188 total damaged building clusters found
- 268 total intact building clusters found
- Spot-checked Area_1/pp1: 4 damaged + 9 intact = 13 total (visually ~12, close enough)

### 3. Clustering Validation
- Tested multiple DBSCAN eps values (3.0, 1.0, 0.5, 0.3)
- eps=3.0 merges everything into 1 cluster (too big)
- eps=0.5 with 3D + min cluster filter gives reasonable counts
- Validation images saved in `3DAeroVLM/visualizations/`

**Known limitation:** DBSCAN may be off by 1-2 buildings on some scenes. Acceptable because QA pairs use multiple-choice with +/-20% deviation options.

### How to re-run fact extraction:
```bash
python ground_truth/extract_facts.py --input ../processed_data --output data/scene_facts.json
```

---

## What Needs To Be Done

### Step 1: Spot-Check Clustering Results (Optional but Recommended)
**Time: 1-2 hours**

Pick 5-6 scenes from different areas, render their clusters, and compare to the RGB visualization to see if building counts are in the right ballpark.

Scenes to check (mix of small/large, different areas):
- Area_1/pp1 (already checked — 13 total, looks right)
- Area_2/pp1 (test set)
- Area_4/pp2
- Area_6/pp10 (found 5 buildings — verify)
- Area_7/pp4
- Area_8/pp2

If any scene is wildly off, we can adjust eps per-area or manually correct the JSON.

---

### Step 2: Build Instruction Pair Generator
**Time: ~1 week | Directory: `instruction_gen/`**

This is the core of the project. For each scene, generate 10-15 question-answer pairs across 5 task types, following DisasterM3's taxonomy.

#### Task Type 1: Scene Recognition (2-3 questions per scene)
Uses: class_percentages, has_road, has_trees from scene_facts.json

Example questions:
- "What structures are present in this 3D scene?"
- "What types of objects can be identified in this point cloud?"
- "Describe the land cover types visible in this disaster scene."

Example answer: "The scene contains damaged buildings (23.5%), intact buildings (15.2%), roads (8.1%), trees (31.4%), and background terrain (21.8%)."

Ground truth source: Directly from class_percentages in scene_facts.json

#### Task Type 2: Damage Counting (2-3 questions per scene)
Uses: num_damaged_buildings, num_intact_buildings, total_buildings, damage_ratio

Example questions:
- "How many damaged buildings are in this 3D scene?" (multiple choice)
- "How many intact buildings remain?" (multiple choice)
- "What is the ratio of damaged to total buildings?" (multiple choice)

Multiple-choice option generation (following DisasterM3):
- Correct answer + 4 wrong options at ±20% and ±40% deviation
- If correct answer is small (< 5), use ±1 and ±2 instead

Ground truth source: DBSCAN cluster counts from scene_facts.json

#### Task Type 3: 3D Referring Segmentation (1-2 questions per scene)
Uses: per-building centroids, bounding boxes from scene_facts.json + original labels

Example questions:
- "Identify the points belonging to damaged buildings."
- "Segment the intact buildings in this scene."
- "Which points represent road surfaces?"

Answer: Point-level mask (list of point indices with the target label). For the benchmark, the ground truth is simply the label mask from the original annotations.

Ground truth source: Labels directly from PLY files (label == 1 for damaged, etc.)

#### Task Type 4: 3D Spatial Reasoning (2-3 questions per scene)
Uses: spatial relationships, building centroids, heights from scene_facts.json

Example questions:
- "What is the distance between the nearest damaged and intact building?"
- "Describe the spatial arrangement of damaged buildings in the scene."
- "Which damaged building is closest to the road?"
- "What is the height of the tallest damaged structure?"

Example answer: "The nearest damaged building is approximately 4.2 meters from the closest intact building. The damaged buildings are clustered in the northern portion of the scene."

Ground truth source: Spatial distances and centroids from scene_facts.json

#### Task Type 5: Damage Report Generation (1-2 questions per scene)
Uses: All facts combined

Example questions:
- "Generate a comprehensive damage assessment report for this 3D scene."
- "Summarize the structural damage visible in this point cloud."

Example answer: "This scene contains 4 damaged buildings and 9 intact buildings (31% damage ratio). The tallest damaged structure is 8.2m. Damaged buildings are concentrated in the eastern portion, approximately 3.5m from the main road. Road infrastructure is present and appears intact. Tree coverage is moderate at 12.2% of the scene."

Ground truth source: Composite of all scene_facts.json fields

#### Question Variation Generation
Following DisasterM3's approach:
- Use Claude API to generate 3-4 phrasings of each base question
- This creates diversity in how models are tested
- Example: "How many damaged buildings?" / "Count the destroyed structures" / "What is the number of buildings showing damage?"

#### Output Format
```json
{
  "scene_id": "Area_1/pp1.ply",
  "split": "train",
  "task_type": "counting",
  "question": "How many damaged buildings are in this 3D scene?",
  "options": ["A. 2", "B. 4", "C. 6", "D. 3", "E. 5"],
  "answer": "B",
  "ground_truth_value": 4,
  "modality": "3D_point_cloud"
}
```

#### Split Assignment
- **Train (Instruct set):** Areas 1, 3, 4, 5, 6, 7, 8 — 56 scenes
- **Test (Bench set):** Area 2 — 8 scenes
- (Same split as 3DAeroRelief paper)

#### Target Counts
- 64 scenes x ~15 pairs per scene = **~960 instruction pairs**
- Split: ~840 train, ~120 test

---

### Step 3: Generate the Instruction Pairs
**Time: 2-3 days | Output: `data/`**

1. Write `instruction_gen/generate_pairs.py` — reads scene_facts.json, generates all pairs
2. For question variations, use Claude API (Anthropic Python SDK)
3. For multiple-choice options, use controlled deviation (±20%, ±40%)
4. Save outputs:
   - `data/3daero_vlm_instruct.json` — training set (Areas 1, 3-8)
   - `data/3daero_vlm_bench.json` — test set (Area 2)
   - `data/3daero_vlm_all.json` — combined

#### Claude API Setup
```bash
pip install anthropic
export ANTHROPIC_API_KEY="your-key-here"
```

```python
import anthropic
client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=500,
    messages=[{
        "role": "user",
        "content": f"Generate 3 different ways to ask: How many damaged buildings are in this scene? Keep each under 15 words."
    }]
)
```

---

### Step 4: Benchmark 3D VLMs
**Time: 1-2 weeks | Directory: `evaluation/`**

#### Candidate Models (discuss with Prof. Rahnemoonfar)
- **3D-LLM** — takes 3D features + language, generates text answers
- **LL3DA** — 3D large language model assistant
- **LEO** — 3D embodied language model
- **Chat-3D v2** — 3D scene dialogue

These are indoor-focused models (ScanNet, ScanQA) — adapting to outdoor disaster scenes IS part of the contribution.

#### What to Build
1. **Data loader** — convert our PLY format to each model's expected input format
2. **Evaluation script** — run each model on bench set, compute accuracy
3. **Baseline comparison** — compare 3D VLM results vs 3DAeroRelief's segmentation baselines

#### Evaluation Metrics (following DisasterM3)
- Multiple-choice QA tasks: Accuracy (%)
- Report generation: GPT-4 scoring on Damage Assessment Precision, Detail Recall, Factual Correctness (scale 0-5)
- Referring segmentation: mIoU, cIoU

---

### Step 5: Analysis and Paper
**Time: 1 week**

1. **Results tables** — model performance across all task types
2. **3D vs 2D analysis** — show what 3D enables that 2D can't (height queries, volume, occlusion)
3. **Failure analysis** — where do 3D VLMs struggle? (likely counting, same as DisasterM3 found)
4. **Visualizations** — example scenes with predictions vs ground truth

Paper title: "3DAeroVLM: A 3D Vision-Language Benchmark for UAV-Based Post-Disaster Assessment"

---

## Key Files Quick Reference

| File | What It Does | How to Run |
|------|-------------|------------|
| `ground_truth/extract_facts.py` | Extracts building counts, heights, spatial info | `python ground_truth/extract_facts.py --input ../processed_data --output data/scene_facts.json` |
| `data/scene_facts.json` | All extracted facts (64 scenes) | Auto-generated by above |
| `instruction_gen/generate_pairs.py` | Generates QA instruction pairs | TODO |
| `data/3daero_vlm_instruct.json` | Training instruction pairs | TODO |
| `data/3daero_vlm_bench.json` | Test instruction pairs | TODO |

---

## Important Notes

- **Always activate venv:** `source /media/volume/Tene_Volume/venv/bin/activate`
- **Large scenes:** Area_3 and Area_5 have scenes with 5-16M points. Scripts subsample to 50K for clustering.
- **DBSCAN params:** eps=0.5, min_samples=20, min_cluster_points=200. These work well but may need per-scene tuning.
- **Train/Test split:** Area 2 = test, everything else = train (same as 3DAeroRelief paper).
- **Delete the zip** if you haven't: `rm /media/volume/Tene_Volume/3DAeroRelief_Dataset.zip` (saves 20GB)

---

## Questions for Prof. Rahnemoonfar

1. Which 3D VLMs to benchmark? (3D-LLM, LL3DA, LEO, Chat-3D?)
2. Should we expand beyond 64 scenes? (Would need new UAV data + COLMAP)
3. Manual annotation or fully automated instruction pair generation?
4. Target venue for paper? (Workshop vs conference)
5. Does the lab have compute for fine-tuning VLMs, or just zero-shot evaluation?
