"""
Ground Truth Fact Extraction from 3DAeroRelief Point Clouds.

For each scene, extracts:
- Building counts (damaged, intact) via DBSCAN clustering
- Building heights, volumes, bounding boxes
- Spatial relationships (distances between buildings, proximity to roads)
- Scene-level statistics (damage ratio, class distributions)

Output: JSON file with per-scene facts used to generate VLM instruction pairs.
"""

import numpy as np
import json
import argparse
from pathlib import Path
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull

# --- Constants ---
LABEL_NAMES = {
    0: 'Background',
    1: 'Building-Damage',
    2: 'Building-No-Damage',
    3: 'Road',
    4: 'Tree'
}


def read_labeled_ply(path):
    """Read a processed PLY file with x,y,z,r,g,b,label fields."""
    with open(path, 'rb') as f:
        n_points = 0
        while True:
            line = f.readline().decode('ascii', errors='ignore').strip()
            if line.startswith('element vertex'):
                n_points = int(line.split()[-1])
            if line == 'end_header':
                break
        dt = np.dtype([
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('r', 'u1'), ('g', 'u1'), ('b', 'u1'),
            ('label', 'i4')
        ])
        data = np.frombuffer(f.read(n_points * dt.itemsize), dtype=dt)
    return data


def cluster_buildings(points, eps=0.5, min_samples=20, max_points=50000):
    """Cluster building points into individual buildings using 3D DBSCAN.

    Uses full 3D (x,y,z) clustering to separate buildings at different heights.
    Subsamples to max_points for memory safety.
    Adaptive min cluster size: requires each cluster to be at least 0.5% of
    the subsampled points (floors at 30, caps at 200).
    Returns list of subsampled point arrays, one per building.
    """
    if len(points) < min_samples:
        return []

    # Subsample for safety
    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        pts = points[idx]
    else:
        pts = points

    # Adaptive threshold: 0.3% of subsampled points, clamped to [30, 150]
    min_cluster_points = max(30, min(150, int(len(pts) * 0.003)))

    # Cluster in 3D to use height differences
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(pts[:, :3])

    buildings = []
    for label_id in set(clustering.labels_):
        if label_id == -1:
            continue
        cluster_pts = pts[clustering.labels_ == label_id]
        if len(cluster_pts) >= min_cluster_points:
            buildings.append(cluster_pts)

    return buildings


def compute_building_stats(building_points):
    """Compute stats for a single building cluster."""
    x, y, z = building_points[:, 0], building_points[:, 1], building_points[:, 2]

    stats = {
        'num_points': len(building_points),
        'centroid': [float(x.mean()), float(y.mean()), float(z.mean())],
        'height': float(z.max() - z.min()),
        'bbox': {
            'x_min': float(x.min()), 'x_max': float(x.max()),
            'y_min': float(y.min()), 'y_max': float(y.max()),
            'z_min': float(z.min()), 'z_max': float(z.max()),
        },
        'footprint_area_approx': float((x.max() - x.min()) * (y.max() - y.min())),
    }

    # Convex hull volume (subsample if too many points)
    if len(building_points) >= 10:
        try:
            pts = building_points[:, :3]
            if len(pts) > 10000:
                pts = pts[np.random.choice(len(pts), 10000, replace=False)]
            hull = ConvexHull(pts)
            stats['volume_approx'] = float(hull.volume)
        except Exception:
            stats['volume_approx'] = None
    else:
        stats['volume_approx'] = None

    return stats


def compute_spatial_relationships(damaged_buildings, intact_buildings, road_points):
    """Compute distances between building clusters and roads."""
    relationships = {}

    # Closest damaged-to-intact distance
    if damaged_buildings and intact_buildings:
        min_dist = float('inf')
        for db in damaged_buildings:
            db_centroid = np.array(db[:, :2].mean(axis=0))
            for ib in intact_buildings:
                ib_centroid = np.array(ib[:, :2].mean(axis=0))
                dist = np.linalg.norm(db_centroid - ib_centroid)
                min_dist = min(min_dist, dist)
        relationships['min_damaged_to_intact_dist'] = float(min_dist)

    # Average distance between damaged buildings
    if len(damaged_buildings) >= 2:
        centroids = np.array([b[:, :2].mean(axis=0) for b in damaged_buildings])
        dists = []
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                dists.append(np.linalg.norm(centroids[i] - centroids[j]))
        relationships['avg_inter_damaged_dist'] = float(np.mean(dists))

    # Closest building to road
    if road_points is not None and len(road_points) > 0:
        road_center = road_points[:, :2].mean(axis=0)
        all_buildings = damaged_buildings + intact_buildings
        if all_buildings:
            dists_to_road = []
            for b in all_buildings:
                b_centroid = b[:, :2].mean(axis=0)
                dists_to_road.append(float(np.linalg.norm(b_centroid - road_center)))
            relationships['min_building_to_road_dist'] = float(min(dists_to_road))
            relationships['avg_building_to_road_dist'] = float(np.mean(dists_to_road))

    return relationships


def extract_scene_facts(ply_path):
    """Extract all ground truth facts from a single scene."""
    data = read_labeled_ply(ply_path)

    points = np.stack([data['x'], data['y'], data['z']], axis=1)
    labels = data['label']

    total_points = len(data)

    # Class distributions
    class_counts = {}
    class_percentages = {}
    for label_id, name in LABEL_NAMES.items():
        count = int(np.sum(labels == label_id))
        class_counts[name] = count
        class_percentages[name] = round(100 * count / total_points, 1)

    # Extract points by class
    damaged_pts = points[labels == 1]
    intact_pts = points[labels == 2]
    road_pts = points[labels == 3]
    tree_pts = points[labels == 4]

    # Cluster into individual buildings
    damaged_buildings = cluster_buildings(damaged_pts)
    intact_buildings = cluster_buildings(intact_pts)

    # Per-building stats
    damaged_stats = [compute_building_stats(b) for b in damaged_buildings]
    intact_stats = [compute_building_stats(b) for b in intact_buildings]

    # Scene-level building summaries
    scene_facts = {
        'scene_id': str(ply_path.relative_to(ply_path.parent.parent)),
        'total_points': total_points,
        'class_counts': class_counts,
        'class_percentages': class_percentages,

        # Building counts
        'num_damaged_buildings': len(damaged_buildings),
        'num_intact_buildings': len(intact_buildings),
        'total_buildings': len(damaged_buildings) + len(intact_buildings),
        'damage_ratio': round(len(damaged_buildings) / max(1, len(damaged_buildings) + len(intact_buildings)), 2),

        # Height stats
        'scene_z_range': float(points[:, 2].max() - points[:, 2].min()),
    }

    if damaged_stats:
        heights = [s['height'] for s in damaged_stats]
        scene_facts['damaged_avg_height'] = round(float(np.mean(heights)), 1)
        scene_facts['damaged_max_height'] = round(float(np.max(heights)), 1)
        scene_facts['damaged_min_height'] = round(float(np.min(heights)), 1)
        scene_facts['tallest_damaged_building'] = damaged_stats[int(np.argmax(heights))]

    if intact_stats:
        heights = [s['height'] for s in intact_stats]
        scene_facts['intact_avg_height'] = round(float(np.mean(heights)), 1)
        scene_facts['intact_max_height'] = round(float(np.max(heights)), 1)

    # Spatial relationships
    scene_facts['spatial'] = compute_spatial_relationships(
        damaged_buildings, intact_buildings, road_pts if len(road_pts) > 0 else None
    )

    # Per-building details
    scene_facts['damaged_buildings'] = damaged_stats
    scene_facts['intact_buildings'] = intact_stats

    # Road and tree coverage
    scene_facts['has_road'] = len(road_pts) > 0
    scene_facts['has_trees'] = len(tree_pts) > 0

    return scene_facts


def main():
    parser = argparse.ArgumentParser(description="Extract ground truth facts from 3DAeroRelief point clouds.")
    parser.add_argument("--input", type=str, required=True, help="Path to processed_data directory")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    args = parser.parse_args()

    input_root = Path(args.input)
    all_facts = []

    # Find all processed PLY files
    scenes = sorted([
        p for p in input_root.rglob('pp*.ply')
        if '_vis_' not in p.name
    ])

    print(f"Found {len(scenes)} scenes to process")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for i, ply_path in enumerate(scenes):
        print(f"[{i+1}/{len(scenes)}] Processing {ply_path.parent.name}/{ply_path.name}...", flush=True)
        try:
            facts = extract_scene_facts(ply_path)
            all_facts.append(facts)
            print(f"  → {facts['num_damaged_buildings']} damaged, {facts['num_intact_buildings']} intact buildings", flush=True)
        except Exception as e:
            print(f"  → ERROR: {e}", flush=True)

        # Save after every scene so we don't lose progress
        with open(output_path, 'w') as f:
            json.dump(all_facts, f, indent=2)

    print(f"\nSaved facts for {len(all_facts)} scenes to {output_path}")


if __name__ == '__main__':
    main()
