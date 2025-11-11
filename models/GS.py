import os, re, glob, math, json
from typing import List, Tuple, Dict, Optional
import numpy as np
import torch
import torch.nn.functional as F
from gs_utlis import *
from gs_render_v2 import *

class KRadarDataset:
    """
    Expects a structure such as:
      root/
        sequences/SEQ_XXXX/
          rdr_sparse/*.npy (or .npz)
          lidar/*.npy|.bin|.npz
    If your layout differs, pass globs via config.
    """
    def __init__(
        self,
        root: str, # "./data/"
        seq_ids: List[str],
        radar_glob: str = "rtnh_wider_1p_1",
        lidar_glob: str = "os2-64/*",
        # Column mapping in rdr_sparse array:
        radar_cols: Dict[str,int] = {"x":0,"y":1,"z":2,"pw":3},  # adjust if needed
        # LiDAR which columns to keep:
        lidar_xyz_cols: Tuple[int,int,int] = (0,1,2),
        # optional filtering:
        max_range_m: Optional[float] = None,
        min_points: int = 16
    ):
        self.root = root
        self.seq_ids = seq_ids
        self.radar_cols = radar_cols
        self.lidar_xyz_cols = lidar_xyz_cols
        self.max_range_m = max_range_m
        self.min_points = min_points

        self.pairs = []  # list of (radar_path, lidar_path, ts_r, ts_l, seq)
        for seq in seq_ids:
            seq_dir = os.path.join(root, "sequences", seq)
            # rdr_sp_dir = os.path.join(root, "rdr_sparse_data/sparse_radar_tensor_wide_range")
            rdr_sp_dir = os.path.join(root, "sequences/gen_sparse_rdr")
            rad_paths = sorted(glob.glob(os.path.join(rdr_sp_dir, radar_glob, seq, '*')))
            lid_paths = sorted(glob.glob(os.path.join(seq_dir, lidar_glob)))
            rad_ts = [(p, find_timestamp(p)) for p in rad_paths]
            lid_ts = [(p, find_timestamp(p)) for p in lid_paths]
            lid_ts = [t for t in lid_ts if t[1] is not None]
            if not rad_ts or not lid_ts:
                continue

            lid_times = np.array([t for _, t in lid_ts], dtype=np.int64)
            lid_paths_only = [p for p,_ in lid_ts]

            for rp, rt in rad_ts:
                if rt is None: continue
                # nearest LiDAR by timestamp
                idx = int(np.argmin(np.abs(lid_times - rt)))
                lp, lt = lid_paths_only[idx], int(lid_times[idx])
                self.pairs.append((rp, lp, rt, lt, seq))
            print(f'dataset len: {len(self.pairs)}')

    def __len__(self): return len(self.pairs)

    def __getitem__(self, i):
        rp, lp, rt, lt, seq = self.pairs[i]
        r_arr = safe_load(rp)
        l_arr = safe_load(lp)

        # Radar XYZ (+ optional power)
        rx, ry, rz = self.radar_cols["x"], self.radar_cols["y"], self.radar_cols["z"]
        radar_xyz = r_arr[:, [rx, ry, rz]].astype(np.float32)
        radar_pw = r_arr[:, self.radar_cols["pw"]].astype(np.float32) if "pw" in self.radar_cols and self.radar_cols["pw"] < r_arr.shape[1] else None

        # Range filtering (optional)
        if self.max_range_m is not None:
            r2 = np.sum(radar_xyz**2, axis=1)
            mask = r2 <= (self.max_range_m**2)
            radar_xyz = radar_xyz[mask]
            if radar_pw is not None:
                radar_pw = radar_pw[mask]

        # LiDAR XYZ
        lx, ly, lz = self.lidar_xyz_cols
        if l_arr.shape[1] <= max(lx,ly,lz):
            raise RuntimeError(f"LiDAR file {lp} has insufficient columns. Got shape {l_arr.shape}")
        lidar_xyz = l_arr[:, [lx, ly, lz]].astype(np.float32)

        # Minimum points sanity check
        if radar_xyz.shape[0] < self.min_points or lidar_xyz.shape[0] < self.min_points:
            # Return empty sample; caller can skip
            return None

        meta = {"radar_path": rp, "lidar_path": lp, "ts_radar": rt, "ts_lidar": lt, "seq": seq}
        return radar_xyz, radar_pw, lidar_xyz, meta

# =====================
# Training per frame or multi-frame
# =====================
def train_gs_on_frame(
    radar_xyz: np.ndarray,
    lidar_xyz: np.ndarray,
    radar_pw: Optional[np.ndarray] = None,
    bounds: Optional[Tuple[Tuple[float,float,float],Tuple[float,float,float]]] = None,
    resolution=(128,128,48),
    init_sigma=0.35,
    epochs=120,
    lr=1e-3,
    lidar_blur_sigma_vox=0.8,
    chunk_size = 4096,
    device="cuda" if torch.cuda.is_available() else "cpu",
    verbose=True
):
    radar_xyz = torch.as_tensor(radar_xyz, device=device)
    lidar_xyz = torch.as_tensor(lidar_xyz, device=device)
    radar_pw_t = torch.as_tensor(radar_pw, device=device) if radar_pw is not None else None

    # Auto bounds if not provided (pad a bit)
    if bounds is None:
        mins = torch.min(torch.vstack([radar_xyz, lidar_xyz]), dim=0).values - torch.tensor([0.5,0.5,0.5], device=device)
        maxs = torch.max(torch.vstack([radar_xyz, lidar_xyz]), dim=0).values + torch.tensor([0.5,0.5,0.5], device=device)
        bounds = (tuple(mins.tolist()), tuple(maxs.tolist()))

    model = RadarGaussianField(radar_xyz, radar_power=radar_pw_t, init_sigma=init_sigma, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    grid_xyz = make_grid(bounds, resolution, device)
    with torch.no_grad():
        lidar_occ = rasterize_points_to_grid_os2(lidar_xyz, bounds, resolution, 
                                                 sigma_vox=lidar_blur_sigma_vox, 
                                                 chunk_size=chunk_size,
                                                 device=device)

    for ep in range(1, epochs+1):
        opt.zero_grad()
        mu, sigma, opacity = model()
        # pred_occ = render_occupancy_voxels(mu, sigma, opacity, grid_xyz)
        pred_occ = render_occupancy_voxels_local_diff(
            mu, sigma, opacity, grid_xyz,
            cutoff_k=3.0,
            max_splats_per_chunk=chunk_size,
            use_autocast=True  # halves memory on H100
        )
        bce = F.binary_cross_entropy(pred_occ, lidar_occ)

        offset_reg = (model.delta**2).mean()
        size_reg = (torch.exp(model.log_sigma).mean())
        opacity_reg = torch.mean(torch.sigmoid(model.logit_opacity))
        loss = bce + 1e-3*offset_reg + 1e-3*size_reg + 1e-3*opacity_reg

        loss.backward()
        opt.step()

        if verbose and (ep % 20 == 0 or ep == 1):
            iou = iou_occupancy(pred_occ.detach(), lidar_occ)
            # metric = compute_metrics(pred_occ.detach(), lidar_occ, thr=0.5)
            metric = metrics_radar_splat_style(pred_occ.detach(), lidar_occ, grid_xyz, thr_pred=0.1)
            print(f"[{ep:04d}] loss={loss.item():.4f}  bce={bce.item():.4f}  \
                  IoU={iou:.3f}, Precision={metric['precision']:.3f}, Recall={metric['recall']:.3f}, F1={metric['f1']:.3f}")

    return model, bounds, resolution

def train_over_dataset(
    dataset: KRadarDataset,
    frames: Optional[int] = None,
    chunk_size = 4096,
    **gs_kwargs
):
    """
    Iterate over frames in the dataset and fit a GS field per frame.
    Returns a list of (model, meta, bounds, resolution).
    """
    out = []
    n = len(dataset) if frames is None else min(frames, len(dataset))
    for i in range(n):
        sample = dataset[i]
        if sample is None:
            continue
        radar_xyz, radar_pw, lidar_xyz, meta = sample
        model, bounds, resolution = train_gs_on_frame(
            radar_xyz=radar_xyz,
            lidar_xyz=lidar_xyz,
            radar_pw=radar_pw,
            chunk_size=chunk_size,
            **gs_kwargs
        )
        out.append((model, meta, bounds, resolution))

        model.eval()
        mu, sigma, opacity = model()
        # pred_occ = render_occupancy_voxels(mu, sigma, opacity, grid_xyz)

        device="cuda" if torch.cuda.is_available() else "cpu"
        radar_xyz = torch.as_tensor(radar_xyz, device=device)
        lidar_xyz = torch.as_tensor(lidar_xyz, device=device)

        mins = torch.min(torch.vstack([radar_xyz, lidar_xyz]), dim=0).values - torch.tensor([0.5,0.5,0.5], device=device)
        maxs = torch.max(torch.vstack([radar_xyz, lidar_xyz]), dim=0).values + torch.tensor([0.5,0.5,0.5], device=device)
        bounds = (tuple(mins.tolist()), tuple(maxs.tolist()))

        grid_xyz = make_grid(bounds, resolution, device)
        pred_occ = render_occupancy_voxels_local_diff(
            mu, sigma, opacity, grid_xyz,
            cutoff_k=3.0,
            max_splats_per_chunk=chunk_size,
            use_autocast=True  # halves memory on H100
        )
        voxel_size_m, grid_origin_m = derive_grid_params_from_grid_xyz(grid_xyz.permute(1,2,3,0))
        # peaks = nms_localmax_nd(pred_occ, radius_vox=2, min_conf=0.3)
        # pts = extract_centroids_metric(pred_occ, peaks, voxel_size_m, grid_origin_m)
        splt_pts = occupancy_to_sparse_points(pred_occ,
                    voxel_size_m=voxel_size_m,
                    grid_origin_m=grid_origin_m,
                    nms_radius_vox=1,
                    min_conf=0.05,
                    window_radius=2,
                    score_mode="integral")
        root = '/home/juntingd/research/3DImage/lidar/K-radar/data/rdr_splt_data'
        splt_path = os.path.join(root, meta['seq'], meta['radar_path'].split('/')[-1])
        np.save(splt_path, splt_pts.detach().cpu().numpy())
    return out

# =====================
# Example CLI
# =====================
if __name__ == "__main__":
    """
    Example usage:
      python kradar_gs.py \
        --root /path/to/KRadar \
        --seq SEQ_0001 SEQ_0002 \
        --frames 2
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default='./data', type=str, help="K-Radar root dir")
    parser.add_argument("--seq", default=['1'], nargs="+", help="Sequence IDs, e.g., SEQ_0001")
    parser.add_argument("--frames", default=None, type=int, help="How many paired frames to train")
    parser.add_argument("--max_range_m", type=float, default=None)
    parser.add_argument("--resolution", type=str, default="64, 64, 32")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--init_sigma", type=float, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lidar_blur_sigma_vox", type=float, default=0.8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk_size", type=int, default=50000)
    args = parser.parse_args()

    res = tuple(int(x) for x in args.resolution.split(","))
    ds = KRadarDataset(
        root=args.root,
        seq_ids=args.seq,
        radar_glob = "rtnh_wider_01p_1/",
        lidar_glob = "os2-64/*",
        radar_cols={"x":0,"y":1,"z":2,"pw":3},  # change if your rdr_sparse layout differs
        lidar_xyz_cols=(0,1,2),
        max_range_m=args.max_range_m,
        min_points=16
    )

    results = train_over_dataset(
        dataset=ds,
        frames=args.frames,
        resolution=res,
        epochs=args.epochs,
        init_sigma=args.init_sigma,
        lr=args.lr,
        lidar_blur_sigma_vox=args.lidar_blur_sigma_vox,
        chunk_size=args.chunk_size,
        device=args.device,
        verbose=True,
    )
    
    # Print a quick summary
    for (model, meta, bounds, resolution) in results:
        with torch.no_grad():
            mu, sigma, opacity = model()
            print(f"Trained frame: {os.path.basename(meta['radar_path'])} | seq={meta['seq']} | "
                  f"radar_pts={mu.shape[0]} | bounds={bounds} | res={resolution}")