"""
Generate VLM instruction pairs from scene facts.

Follows DisasterM3's task taxonomy adapted for 3D point clouds:
1. Scene Recognition
2. Damage Counting
3. Referring Segmentation
4. Spatial Reasoning
5. Damage Report Generation

Usage:
    python instruction_gen/generate_pairs.py \
        --facts data/scene_facts.json \
        --output-train data/3daero_vlm_instruct.json \
        --output-test data/3daero_vlm_bench.json \
        --output-all data/3daero_vlm_all.json
"""

import json
import random
import argparse
from pathlib import Path

# Area 2 = test set, everything else = train (same as 3DAeroRelief paper)
TEST_AREAS = {"Area_2"}

random.seed(42)


# ──────────────────────────────────────────────
# Multiple-choice option generation
# ──────────────────────────────────────────────

def generate_count_options(correct, num_options=5):
    """Generate multiple-choice options for counting questions.
    Following DisasterM3: ±20% and ±40% deviations.
    For small numbers (< 5), use ±1 and ±2.
    """
    if correct < 5:
        candidates = {correct - 2, correct - 1, correct, correct + 1, correct + 2}
        candidates = {max(0, c) for c in candidates}
        candidates.discard(correct)
        wrong = sorted(candidates)
        while len(wrong) < num_options - 1:
            wrong.append(correct + len(wrong) + 1)
        wrong = wrong[:num_options - 1]
    else:
        deviations = [0.2, 0.4, -0.2, -0.4]
        wrong = []
        for d in deviations:
            val = max(0, int(round(correct * (1 + d))))
            if val != correct and val not in wrong:
                wrong.append(val)
        # Fill if needed
        offset = 1
        while len(wrong) < num_options - 1:
            candidate = correct + offset
            if candidate not in wrong and candidate != correct:
                wrong.append(candidate)
            offset = -offset if offset > 0 else -offset + 1

    options = [correct] + wrong[:num_options - 1]
    random.shuffle(options)
    correct_idx = options.index(correct)
    correct_letter = chr(65 + correct_idx)

    formatted = [f"{chr(65 + i)}. {v}" for i, v in enumerate(options)]
    return formatted, correct_letter


def generate_ratio_options(correct_pct, num_options=5):
    """Generate multiple-choice options for percentage/ratio questions."""
    candidates = set()
    for delta in [5, 10, 15, 20, -5, -10, -15, -20]:
        val = max(0, min(100, round(correct_pct + delta)))
        if val != round(correct_pct):
            candidates.add(val)

    wrong = sorted(candidates)[:num_options - 1]
    options = [round(correct_pct)] + wrong
    random.shuffle(options)
    correct_idx = options.index(round(correct_pct))
    correct_letter = chr(65 + correct_idx)

    formatted = [f"{chr(65 + i)}. {v}%" for i, v in enumerate(options)]
    return formatted, correct_letter


def generate_height_options(correct_m, num_options=5):
    """Generate multiple-choice options for height questions."""
    correct_rounded = round(correct_m, 1)
    candidates = set()
    for delta in [2.0, 4.0, -2.0, -4.0, 1.0, -1.0, 3.0, -3.0]:
        val = round(max(0, correct_m + delta), 1)
        if val != correct_rounded:
            candidates.add(val)

    wrong = sorted(candidates)[:num_options - 1]
    options = [correct_rounded] + wrong
    random.shuffle(options)
    correct_idx = options.index(correct_rounded)
    correct_letter = chr(65 + correct_idx)

    formatted = [f"{chr(65 + i)}. {v}m" for i, v in enumerate(options)]
    return formatted, correct_letter


# ──────────────────────────────────────────────
# Task generators
# ──────────────────────────────────────────────

def generate_recognition_pairs(facts):
    """Task 1: Scene Recognition — identify what structures/classes are present."""
    pairs = []
    scene_id = facts['scene_id']
    pcts = facts['class_percentages']

    # Which classes are present (> 1%)?
    present = [name for name, pct in pcts.items() if pct > 1.0 and name != 'Background']
    present_str = ", ".join(present).lower()

    questions = [
        "What types of structures and land cover are present in this 3D scene?",
        "Identify the main object categories visible in this point cloud.",
        "What can you observe in this 3D disaster scene?",
    ]

    # Build descriptive answer
    parts = []
    for name in ['Building-Damage', 'Building-No-Damage', 'Road', 'Tree']:
        if pcts.get(name, 0) > 1.0:
            parts.append(f"{name.lower().replace('-', ' ')} ({pcts[name]}%)")
    answer = f"The scene contains: {', '.join(parts)}. Background makes up {pcts.get('Background', 0)}% of the scene."

    for q in questions:
        pairs.append({
            "scene_id": scene_id,
            "task_type": "scene_recognition",
            "question": q,
            "answer": answer,
            "format": "open_ended",
        })

    # Also a multiple-choice: what is the dominant class?
    non_bg = {k: v for k, v in pcts.items() if k != 'Background'}
    dominant = max(non_bg, key=non_bg.get) if non_bg else 'Background'

    class_options = ['Building-Damage', 'Building-No-Damage', 'Road', 'Tree']
    random.shuffle(class_options)
    correct_idx = class_options.index(dominant)
    correct_letter = chr(65 + correct_idx)
    formatted = [f"{chr(65 + i)}. {c}" for i, c in enumerate(class_options)]

    pairs.append({
        "scene_id": scene_id,
        "task_type": "scene_recognition",
        "question": "What is the most prevalent non-background class in this 3D scene?",
        "options": formatted,
        "answer": correct_letter,
        "ground_truth_value": dominant,
        "format": "multiple_choice",
    })

    return pairs


def generate_counting_pairs(facts):
    """Task 2: Damage Counting — count buildings, compute ratios."""
    pairs = []
    scene_id = facts['scene_id']

    # Q1: How many damaged buildings?
    n_damaged = facts['num_damaged_buildings']
    if n_damaged > 0:
        questions_damaged = [
            "How many damaged buildings are in this 3D scene?",
            "Count the number of destroyed or damaged structures visible in this point cloud.",
            "What is the total count of buildings showing damage?",
        ]
        options, answer = generate_count_options(n_damaged)
        for q in questions_damaged:
            pairs.append({
                "scene_id": scene_id,
                "task_type": "counting",
                "question": q,
                "options": options,
                "answer": answer,
                "ground_truth_value": n_damaged,
                "format": "multiple_choice",
            })

    # Q2: How many intact buildings?
    n_intact = facts['num_intact_buildings']
    if n_intact > 0:
        questions_intact = [
            "How many intact buildings remain in this scene?",
            "Count the undamaged buildings in this 3D point cloud.",
        ]
        options, answer = generate_count_options(n_intact)
        for q in questions_intact:
            pairs.append({
                "scene_id": scene_id,
                "task_type": "counting",
                "question": q,
                "options": options,
                "answer": answer,
                "ground_truth_value": n_intact,
                "format": "multiple_choice",
            })

    # Q3: Total buildings
    total = facts['total_buildings']
    if total > 0:
        options, answer = generate_count_options(total)
        pairs.append({
            "scene_id": scene_id,
            "task_type": "counting",
            "question": "How many total buildings (damaged and intact) are in this 3D scene?",
            "options": options,
            "answer": answer,
            "ground_truth_value": total,
            "format": "multiple_choice",
        })

    # Q4: Damage ratio
    if total > 0:
        ratio_pct = round(facts['damage_ratio'] * 100)
        options, answer = generate_ratio_options(ratio_pct)
        pairs.append({
            "scene_id": scene_id,
            "task_type": "counting",
            "question": "What percentage of buildings in this scene are damaged?",
            "options": options,
            "answer": answer,
            "ground_truth_value": ratio_pct,
            "format": "multiple_choice",
        })

    return pairs


def generate_segmentation_pairs(facts):
    """Task 3: Referring Segmentation — identify which points belong to a class.
    Ground truth is the label mask from the original PLY file.
    """
    pairs = []
    scene_id = facts['scene_id']
    pcts = facts['class_percentages']

    seg_targets = [
        ("Building-Damage", 1, [
            "Segment the damaged buildings in this 3D scene.",
            "Identify all points belonging to damaged structures.",
            "Highlight the destroyed buildings in this point cloud.",
        ]),
        ("Building-No-Damage", 2, [
            "Segment the intact buildings in this 3D scene.",
            "Identify all points belonging to undamaged structures.",
        ]),
        ("Road", 3, [
            "Segment the road surfaces in this 3D scene.",
            "Identify all points that belong to roads.",
        ]),
        ("Tree", 4, [
            "Segment the trees and vegetation in this 3D scene.",
            "Identify all points belonging to trees.",
        ]),
    ]

    for class_name, label_id, questions in seg_targets:
        if pcts.get(class_name, 0) < 0.5:
            continue  # skip classes with negligible presence

        for q in questions:
            pairs.append({
                "scene_id": scene_id,
                "task_type": "referring_segmentation",
                "question": q,
                "answer": f"The {class_name.lower().replace('-', ' ')} points are identified. They comprise {pcts[class_name]}% of the scene ({facts['class_counts'][class_name]:,} points).",
                "ground_truth_label": label_id,
                "ground_truth_class": class_name,
                "format": "segmentation",
            })

    return pairs


def generate_spatial_pairs(facts):
    """Task 4: Spatial Reasoning — distances, heights, arrangement."""
    pairs = []
    scene_id = facts['scene_id']
    spatial = facts.get('spatial', {})

    # Q1: Distance between damaged and intact
    if 'min_damaged_to_intact_dist' in spatial:
        dist = round(spatial['min_damaged_to_intact_dist'], 1)
        pairs.append({
            "scene_id": scene_id,
            "task_type": "spatial_reasoning",
            "question": "What is the distance between the nearest damaged and intact building?",
            "answer": f"The nearest damaged building is approximately {dist} meters from the closest intact building.",
            "ground_truth_value": dist,
            "format": "open_ended",
        })

    # Q2: Average spacing between damaged buildings
    if 'avg_inter_damaged_dist' in spatial:
        avg_dist = round(spatial['avg_inter_damaged_dist'], 1)
        pairs.append({
            "scene_id": scene_id,
            "task_type": "spatial_reasoning",
            "question": "How spread out are the damaged buildings from each other?",
            "answer": f"The damaged buildings are spaced an average of {avg_dist} meters apart.",
            "ground_truth_value": avg_dist,
            "format": "open_ended",
        })

    # Q4: Proximity to road
    if 'min_building_to_road_dist' in spatial:
        road_dist = round(spatial['min_building_to_road_dist'], 1)
        pairs.append({
            "scene_id": scene_id,
            "task_type": "spatial_reasoning",
            "question": "How far is the nearest building from the road?",
            "answer": f"The nearest building is approximately {road_dist} meters from the road.",
            "ground_truth_value": road_dist,
            "format": "open_ended",
        })

    return pairs


def generate_report_pairs(facts):
    """Task 5: Damage Report Generation — comprehensive assessment."""
    pairs = []
    scene_id = facts['scene_id']
    pcts = facts['class_percentages']

    # Build report from all facts
    parts = []

    # Buildings summary
    n_d = facts['num_damaged_buildings']
    n_i = facts['num_intact_buildings']
    total = facts['total_buildings']
    ratio = facts['damage_ratio']

    if total > 0:
        parts.append(f"This scene contains {total} buildings: {n_d} damaged and {n_i} intact ({round(ratio*100)}% damage rate).")

    # Spatial context
    spatial = facts.get('spatial', {})
    if 'min_damaged_to_intact_dist' in spatial:
        parts.append(f"The nearest damaged building is {round(spatial['min_damaged_to_intact_dist'], 1)}m from the closest intact structure.")

    if 'avg_inter_damaged_dist' in spatial:
        parts.append(f"Damaged buildings are spaced {round(spatial['avg_inter_damaged_dist'], 1)}m apart on average.")

    # Infrastructure
    if facts['has_road']:
        parts.append(f"Road infrastructure is present ({pcts.get('Road', 0)}% of scene).")
        if 'min_building_to_road_dist' in spatial:
            parts.append(f"The nearest building is {round(spatial['min_building_to_road_dist'], 1)}m from the road.")

    if facts['has_trees']:
        parts.append(f"Vegetation covers {pcts.get('Tree', 0)}% of the scene.")

    # Scene overview
    parts.append(f"Total scene: {facts['total_points']:,} 3D points.")

    full_report = " ".join(parts)

    questions = [
        "Generate a comprehensive damage assessment report for this 3D scene.",
        "Provide a detailed analysis of the structural damage in this point cloud.",
    ]

    for q in questions:
        pairs.append({
            "scene_id": scene_id,
            "task_type": "damage_report",
            "question": q,
            "answer": full_report,
            "format": "open_ended",
        })

    return pairs


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def generate_all_pairs(facts_list):
    """Generate all instruction pairs for all scenes."""
    all_pairs = []

    for facts in facts_list:
        scene_id = facts['scene_id']
        area = scene_id.split('/')[0]
        split = "test" if area in TEST_AREAS else "train"

        scene_pairs = []
        scene_pairs.extend(generate_recognition_pairs(facts))
        scene_pairs.extend(generate_counting_pairs(facts))
        scene_pairs.extend(generate_segmentation_pairs(facts))
        scene_pairs.extend(generate_spatial_pairs(facts))
        scene_pairs.extend(generate_report_pairs(facts))

        # Add split and modality to all pairs
        for pair in scene_pairs:
            pair['split'] = split
            pair['modality'] = '3D_point_cloud'

        all_pairs.extend(scene_pairs)

    return all_pairs


def main():
    parser = argparse.ArgumentParser(description="Generate VLM instruction pairs from scene facts.")
    parser.add_argument("--facts", type=str, required=True, help="Path to scene_facts.json")
    parser.add_argument("--output-train", type=str, default="data/3daero_vlm_instruct.json")
    parser.add_argument("--output-test", type=str, default="data/3daero_vlm_bench.json")
    parser.add_argument("--output-all", type=str, default="data/3daero_vlm_all.json")
    args = parser.parse_args()

    with open(args.facts) as f:
        facts_list = json.load(f)

    print(f"Loaded facts for {len(facts_list)} scenes")

    all_pairs = generate_all_pairs(facts_list)

    train_pairs = [p for p in all_pairs if p['split'] == 'train']
    test_pairs = [p for p in all_pairs if p['split'] == 'test']

    # Save
    for path, data in [
        (args.output_train, train_pairs),
        (args.output_test, test_pairs),
        (args.output_all, all_pairs),
    ]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(data)} pairs to {path}")

    # Summary
    print(f"\n{'='*50}")
    print(f"Total instruction pairs: {len(all_pairs)}")
    print(f"  Train: {len(train_pairs)}")
    print(f"  Test:  {len(test_pairs)}")
    print()

    # By task type
    from collections import Counter
    task_counts = Counter(p['task_type'] for p in all_pairs)
    print("By task type:")
    for task, count in sorted(task_counts.items()):
        print(f"  {task}: {count}")

    # By format
    format_counts = Counter(p['format'] for p in all_pairs)
    print("\nBy format:")
    for fmt, count in sorted(format_counts.items()):
        print(f"  {fmt}: {count}")


if __name__ == '__main__':
    main()
