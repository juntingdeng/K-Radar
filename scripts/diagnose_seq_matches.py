"""
Standalone diagnostic: for each K-Radar sequence, reports the fraction of frames where
local_match_closest_mdn finds zero radar-LiDAR correspondences within R (the same
"empty-match" condition that makes SynthLocalLoss_MDN fall back to a zero loss for that
batch during training -- ~15% of batches in the seqs 1/13/58 run).

Runs no network and no training -- just data loading, radar/LiDAR preprocessing, and the
matching function -- so it's fast and isolates whether the empty-match rate concentrates
in specific sequences (e.g. poor radar/LiDAR alignment or much sparser point density in a
particular sequence) rather than being spread evenly.

Usage:
    python scripts/diagnose_seq_matches.py --seqs 1 13 58 --search_radius 5.0 --split train
"""
import os, sys
import argparse
import collections

import torch
from torch.utils.data import DataLoader
from spconv.pytorch import SparseConvTensor

# this script lives in scripts/; add the repo root so project imports below resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.util_config import *
from datasets.kradar_detection_v2_0 import KRadarDetection_v2_0
from dataset_utils.KDataset import RadarSparseProcessor, LdrPreprocessor
from models.generatives.generative import local_match_closest_mdn, voxel_axis_scale


def arg_parser():
    args = argparse.ArgumentParser()
    args.add_argument('--seqs', nargs='+', required=True, help='e.g. --seqs 1 13 58')
    args.add_argument('--search_radius', default=5.0, type=float)
    args.add_argument('--split', default='train', choices=['train', 'test'])
    args.add_argument('--stride', default=1, type=int,
                       help='process every Nth frame instead of all of them, e.g. --stride 5 '
                            'for a 5x speedup at a small cost to the rate estimate precision.')
    args.add_argument('--num_workers', default=4, type=int,
                       help='DataLoader workers, so point-cloud file I/O for the next frame '
                            'overlaps with the current frame\'s CPU-bound voxelization/matching.')
    return args.parse_args()


if __name__ == '__main__':
    d = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = arg_parser()

    print(f'device: {d}', flush=True)
    print(f'loading cfg and scanning sequences {args.seqs} (split={args.split}) ...', flush=True)
    cfg_path = './configs/cfg_rdr_ldr.yml'
    cfg = cfg_from_yaml_file(cfg_path, cfg)
    cfg.DATASET.path_data.list_seq = [str(s) for s in args.seqs]

    x_min, y_min, z_min, x_max, y_max, z_max = cfg.DATASET.roi.xyz
    vsize_xyz_list = cfg.DATASET.roi.voxel_size
    x_size = int(round((x_max - x_min) / vsize_xyz_list[0]))
    y_size = int(round((y_max - y_min) / vsize_xyz_list[1]))
    z_size = int(round((z_max - z_min) / vsize_xyz_list[2]))
    vsize_xyz = torch.tensor(vsize_xyz_list).to(d)
    axis_scale = voxel_axis_scale(vsize_xyz)

    kdataset = KRadarDetection_v2_0(cfg=cfg, split=args.split)
    print(f'dataset loaded: {len(kdataset)} frames across seqs {args.seqs}', flush=True)
    if args.stride > 1:
        kdataset = torch.utils.data.Subset(kdataset, range(0, len(kdataset), args.stride))
        kdataset.collate_fn = kdataset.dataset.collate_fn
        print(f'stride={args.stride}: sampling {len(kdataset)} of those frames', flush=True)
    dataloader = DataLoader(kdataset, batch_size=1, collate_fn=kdataset.collate_fn,
                             num_workers=args.num_workers, shuffle=False)

    rdr_processor = RadarSparseProcessor(cfg)
    ldr_processor = LdrPreprocessor(cfg)
    print('starting per-frame matching...', flush=True)

    # per-seq counters: [n_frames, n_empty_match_frames, n_radar_voxels_total, n_matched_voxels_total]
    stats = collections.defaultdict(lambda: [0, 0, 0, 0])

    for bi, batch_dict in enumerate(dataloader):
        seq = batch_dict['meta'][0]['seq']

        batch_dict = rdr_processor.forward(batch_dict)
        batch_dict = ldr_processor.forward(batch_dict)

        for key in ['voxels', 'voxel_coords', 'sp_features', 'sp_indices']:
            val = batch_dict[key]
            if isinstance(val, torch.Tensor) and val.device != d:
                batch_dict[key] = val.to(d)

        radar_st = SparseConvTensor(
            features=batch_dict['sp_features'].reshape((batch_dict['sp_features'].shape[0], -1)),
            indices=batch_dict['sp_indices'].int(),
            spatial_shape=[z_size, y_size, x_size], batch_size=1)
        lidar_st = SparseConvTensor(
            features=batch_dict['voxels'].reshape((batch_dict['voxels'].shape[0], -1)),
            indices=batch_dict['voxel_coords'].int(),
            spatial_shape=[z_size, y_size, x_size], batch_size=1)

        matched_mask, _, _, _, _ = local_match_closest_mdn(
            radar_st, lidar_st, gt_topk=100, R=args.search_radius, axis_scale=axis_scale)

        s = stats[seq]
        s[0] += 1
        s[1] += int(not matched_mask.any().item())
        s[2] += matched_mask.numel()
        s[3] += int(matched_mask.sum().item())

        if bi == 0 or (bi + 1) % 20 == 0:
            print(f'  processed {bi + 1} frames...', flush=True)

    print(f'\n{"seq":>6} {"frames":>8} {"empty-match frames":>20} {"empty-match %":>15} {"voxel match rate":>18}')
    total = [0, 0, 0, 0]
    for seq in sorted(stats.keys(), key=lambda s: int(s)):
        n_frames, n_empty, n_vox, n_matched_vox = stats[seq]
        for i, v in enumerate([n_frames, n_empty, n_vox, n_matched_vox]):
            total[i] += v
        empty_pct = 100 * n_empty / max(1, n_frames)
        voxel_rate = 100 * n_matched_vox / max(1, n_vox)
        print(f'{seq:>6} {n_frames:>8} {n_empty:>20} {empty_pct:>14.1f}% {voxel_rate:>17.1f}%')

    n_frames, n_empty, n_vox, n_matched_vox = total
    print(f'{"all":>6} {n_frames:>8} {n_empty:>20} {100*n_empty/max(1,n_frames):>14.1f}% {100*n_matched_vox/max(1,n_vox):>17.1f}%')
