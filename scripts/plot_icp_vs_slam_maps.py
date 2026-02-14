#!/usr/bin/env python3
"""
Generate ICP map vs SLAM map (Config A / Config B) side-by-side comparison
for every dataset sequence.
Saves: results/sequence_{N}_icp_vs_slam_maps.png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from pathlib import Path
import yaml

RESULTS = Path(__file__).resolve().parent.parent / 'results'

# ICP map sources per sequence (PNG rendered maps from icp_map_builder)
ICP_MAPS = {
    0: RESULTS / 'sequence_0' / 'ICP_seq0.png',
    1: RESULTS / 'sequence_1' / 'icp_seq1.png',
    2: RESULTS / 'sequence_2' / 'icp_seq2.png',
}

# SLAM map PGM sources per sequence and config
SLAM_MAPS = {
    0: {
        'A': RESULTS / 'sequence_0' / 'Config_A' / 'slam_map.pgm',
        'B': RESULTS / 'sequence_0' / 'Config_B' / 'slam_map.pgm',
    },
    1: {
        'A': RESULTS / 'sequence_1' / 'Config_A' / 'slam_map.pgm',
        'B': RESULTS / 'sequence_1' / 'Config_B' / 'slam_map.pgm',
    },
    2: {
        'A': RESULTS / 'sequence_2' / 'Config_A' / 'slam_map.pgm',
        'B': RESULTS / 'sequence_2' / 'Config_B' / 'slam_map.pgm',
    },
}

SLAM_YAMLS = {
    0: {
        'A': RESULTS / 'sequence_0' / 'Config_A' / 'slam_map.yaml',
        'B': RESULTS / 'sequence_0' / 'Config_B' / 'slam_map.yaml',
    },
    1: {
        'A': RESULTS / 'sequence_1' / 'Config_A' / 'slam_map.yaml',
        'B': RESULTS / 'sequence_1' / 'Config_B' / 'slam_map.yaml',
    },
    2: {
        'A': RESULTS / 'sequence_2' / 'Config_A' / 'slam_map.yaml',
        'B': RESULTS / 'sequence_2' / 'Config_B' / 'slam_map.yaml',
    },
}

SEQ_LABELS = {
    0: 'Sequence 0 – Empty Hallway',
    1: 'Sequence 1 – Sharp Turns',
    2: 'Sequence 2 – Smooth Motion',
}


def load_pgm_with_extent(pgm_path, yaml_path):
    """Load a PGM map and return (image, extent) where extent=[xmin,xmax,ymin,ymax]."""
    img = mpimg.imread(str(pgm_path))
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)
    resolution = float(meta['resolution'])
    origin = meta['origin']  # [x, y, theta]
    h, w = img.shape[:2]
    xmin = origin[0]
    xmax = origin[0] + w * resolution
    ymin = origin[1]
    ymax = origin[1] + h * resolution
    return img, [xmin, xmax, ymin, ymax]


def plot_pgm(ax, pgm_path, yaml_path, title, color_map='gray'):
    """Plot a PGM map on an axis with real-world coordinates."""
    if not pgm_path.exists():
        ax.text(0.5, 0.5, 'Map not found\n' + pgm_path.name,
                ha='center', va='center', transform=ax.transAxes, fontsize=9,
                color='red')
        ax.set_title(title, fontsize=11, fontweight='bold')
        return
    img, extent = load_pgm_with_extent(pgm_path, yaml_path)
    # PGM: 205=free(white), 0=occupied(black), 128=unknown(gray)
    ax.imshow(img, cmap=color_map, origin='upper',
              extent=extent, interpolation='nearest', aspect='equal')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('X (m)', fontsize=9)
    ax.set_ylabel('Y (m)', fontsize=9)
    ax.grid(True, alpha=0.2, color='blue', linewidth=0.5)


def plot_png(ax, png_path, title):
    """Plot a PNG map image on an axis."""
    if not png_path.exists():
        ax.text(0.5, 0.5, 'Image not found\n' + png_path.name,
                ha='center', va='center', transform=ax.transAxes, fontsize=9,
                color='red')
        ax.set_title(title, fontsize=11, fontweight='bold')
        return
    img = mpimg.imread(str(png_path))
    ax.imshow(img, aspect='equal')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.axis('off')


def generate_comparison(seq_id):
    """Generate ICP vs SLAM Config A vs SLAM Config B comparison for one sequence."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f'2D Map Comparison – {SEQ_LABELS[seq_id]}\n'
        f'ICP Map  |  SLAM Config A (Relaxed ±15 cm)  |  SLAM Config B (Strict ±1.5 cm)',
        fontsize=13, fontweight='bold', y=1.02
    )

    # Column 0: ICP map (PNG)
    plot_png(axes[0], ICP_MAPS[seq_id],
             'ICP Map\n(LiDAR scans from ICP poses)')

    # Column 1: SLAM Config A map
    plot_pgm(axes[1],
             SLAM_MAPS[seq_id]['A'],
             SLAM_YAMLS[seq_id]['A'],
             'SLAM Map – Config A\n(Relaxed: search ±15 cm, penalty=2.5)')

    # Column 2: SLAM Config B map
    plot_pgm(axes[2],
             SLAM_MAPS[seq_id]['B'],
             SLAM_YAMLS[seq_id]['B'],
             'SLAM Map – Config B\n(Strict: search ±1.5 cm, penalty=20/40)')

    plt.tight_layout()
    out = RESULTS / f'sequence_{seq_id}_icp_vs_slam_maps.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


def generate_all_sequences_grid():
    """Generate a single large grid: rows=sequences, cols=ICP/SLAM-A/SLAM-B."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    fig.suptitle(
        '2D Map Comparison: ICP vs SLAM Config A vs SLAM Config B\n'
        'All Sequences',
        fontsize=15, fontweight='bold', y=1.01
    )

    col_titles = [
        'ICP Map\n(ICP pose-guided)',
        'SLAM Config A\n(Relaxed ±15 cm)',
        'SLAM Config B\n(Strict ±1.5 cm)',
    ]
    row_titles = [SEQ_LABELS[i] for i in range(3)]

    for row, seq_id in enumerate(range(3)):
        for col in range(3):
            ax = axes[row][col]
            if row == 0:
                ax.set_title(col_titles[col], fontsize=11, fontweight='bold', pad=8)
            if col == 0:
                ax.set_ylabel(row_titles[seq_id], fontsize=11,
                              fontweight='bold', labelpad=8)

            if col == 0:
                plot_png(ax, ICP_MAPS[seq_id], '')
            elif col == 1:
                plot_pgm(ax, SLAM_MAPS[seq_id]['A'], SLAM_YAMLS[seq_id]['A'], '')
            else:
                plot_pgm(ax, SLAM_MAPS[seq_id]['B'], SLAM_YAMLS[seq_id]['B'], '')

    plt.tight_layout()
    out = RESULTS / 'all_sequences_icp_vs_slam_maps.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


if __name__ == '__main__':
    for seq in range(3):
        generate_comparison(seq)
    generate_all_sequences_grid()
    print('Done.')
