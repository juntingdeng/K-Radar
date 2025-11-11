import os, re, glob, math, json
from typing import List, Tuple, Dict, Optional
import numpy as np
import torch
import torch.nn.functional as F

from gs_render_v2 import *

# =====================
# Utils: grid + lidar occupancy
# =====================
def make_grid(bounds, resolution, device):
    (xmin, ymin, zmin), (xmax, ymax, zmax) = bounds
    nx, ny, nz = resolution
    xs = torch.linspace(xmin, xmax, nx, device=device)
    ys = torch.linspace(ymin, ymax, ny, device=device)
    zs = torch.linspace(zmin, zmax, nz, device=device)
    grid_z, grid_y, grid_x = torch.meshgrid(zs, ys, xs, indexing="ij")  # [nz,ny,nx]
    grid_xyz = torch.stack([grid_x, grid_y, grid_z], dim=0)  # [3,nz,ny,nx]
    return grid_xyz

def gaussian_kernel1d(sigma, device, radius=None):
    if sigma <= 0:
        # No smoothing → delta kernel
        return torch.tensor([1.0], device=device).view(1, 1, 1)
    r = int(max(2, (3 * sigma) if radius is None else radius))
    x = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = k / k.sum()
    return k.view(1, 1, -1)  # (outC=1, inC=1, K)

def gaussian_blur3d(grid, sigma_vox):
    """
    grid: (N, C, D, H, W)  e.g. (1,1,nz,ny,nx)
    sigma_vox: float (std in 'voxel units')
    """
    device = grid.device
    kx = gaussian_kernel1d(sigma_vox, device)  # shape (1,1,Kx)
    ky = gaussian_kernel1d(sigma_vox, device)  # shape (1,1,Ky)
    kz = gaussian_kernel1d(sigma_vox, device)  # shape (1,1,Kz)

    # Convolve along X (W): kernel (1,1,1,1,Kx), pad on W
    wx = kx.view(1, 1, 1, 1, kx.shape[-1])
    grid = F.conv3d(grid, wx, padding=(0, 0, kx.shape[-1] // 2))

    # Convolve along Y (H): kernel (1,1,1,Ky,1), pad on H
    wy = ky.view(1, 1, 1, ky.shape[-1], 1)
    grid = F.conv3d(grid, wy, padding=(0, ky.shape[-1] // 2, 0))

    # Convolve along Z (D): kernel (1,1,Kz,1,1), pad on D
    wz = kz.view(1, 1, kz.shape[-1], 1, 1)
    grid = F.conv3d(grid, wz, padding=(kz.shape[-1] // 2, 0, 0))

    return grid


def rasterize_points_to_grid(points, bounds, resolution, sigma_vox=1.0, device="cpu"):
    (xmin, ymin, zmin), (xmax, ymax, zmax) = bounds
    nx, ny, nz = resolution
    grid = torch.zeros((1, 1, nz, ny, nx), device=device)

    if points.numel() == 0:
        return grid

    px = (points[:,0] - xmin) / (xmax - xmin) * (nx - 1)
    py = (points[:,1] - ymin) / (ymax - ymin) * (ny - 1)
    pz = (points[:,2] - zmin) / (zmax - zmin) * (nz - 1)
    ix = px.round().clamp(0, nx-1).long()
    iy = py.round().clamp(0, ny-1).long()
    iz = pz.round().clamp(0, nz-1).long()

    grid[0, 0, iz, iy, ix] = 1.0

    if sigma_vox and sigma_vox > 0:
        grid = gaussian_blur3d(grid, sigma_vox).clamp(0, 1)

    return grid

import torch

def os2_sigmas_xyz(points_xyz, vx, vy, vz,
                   sigma_range=0.03,        # m
                   sigma_ang=6.7e-4 ):   # rad
    # Decompose into spherical coordinates
    x, y, z = points_xyz[:,0], points_xyz[:,1], points_xyz[:,2]
    r = torch.sqrt(x**2 + y**2 + z**2).clamp_min(1e-3)
    th = torch.atan2(y, x)                 # azimuth
    ph = torch.asin(z / r)                 # elevation

    # Radial / angular spreads
    sig_rad = torch.full_like(r, sigma_range)
    sig_tan1 = r * sigma_ang            # azimuthal (horizontal)
    sig_tan2 = r * sigma_ang            # elevation (vertical)

    # Convert to x, y, z components (world space)
    sig_x = torch.sqrt((sig_rad*torch.cos(ph)*torch.cos(th))**2 +
                       (sig_tan1*torch.sin(th))**2 +
                       (sig_tan2*torch.sin(ph)*torch.cos(th))**2)

    sig_y = torch.sqrt((sig_rad*torch.cos(ph)*torch.sin(th))**2 +
                       (sig_tan1*torch.cos(th))**2 +
                       (sig_tan2*torch.sin(ph)*torch.sin(th))**2)

    sig_z = torch.sqrt((sig_rad*torch.sin(ph))**2 +
                       (sig_tan2*torch.cos(ph))**2)

    # Convert to voxel scale
    sigx_v = (sig_x / vx).clamp(0.3*vx, 3.0*vx)
    sigy_v = (sig_y / vy).clamp(0.3*vy, 3.0*vy)
    sigz_v = (sig_z / vz).clamp(0.3*vz, 3.0*vz)

    # Representative (median) σ per axis
    return torch.stack([sigx_v, sigy_v, sigz_v], dim=1)

def rasterize_points_to_grid_os2(points, bounds, resolution, chunk_size=4096, sigma_vox=1.0, device="cpu"):
    (xmin, ymin, zmin), (xmax, ymax, zmax) = bounds
    nx, ny, nz = resolution

    vx = (xmax - xmin)/nx
    vy = (ymax - ymin)/ny
    vz = (zmax - zmin)/nz

    px = (points[:,0] - xmin) / (xmax - xmin) * (nx - 1)
    py = (points[:,1] - ymin) / (ymax - ymin) * (ny - 1)
    pz = (points[:,2] - zmin) / (zmax - zmin) * (nz - 1)
    ix = px.round().clamp(0, nx-1).long()
    iy = py.round().clamp(0, ny-1).long()
    iz = pz.round().clamp(0, nz-1).long()

    grid_xyz = make_grid(bounds, resolution, device)

    sigma_vox = os2_sigmas_xyz(points, vx, vy, vz)
    opacity = torch.ones((points.shape[0], 1))
    occ = render_occupancy_voxels_local_diff(points, sigma_vox, opacity=opacity, grid_xyz=grid_xyz, max_splats_per_chunk=chunk_size)
    return occ



def iou_occupancy(pred, tgt, thr=0.5):
    p = (pred >= thr).float()
    t = (tgt >= thr).float()
    inter = (p*t).sum()
    union = (p + t - p*t).sum() + 1e-6
    return (inter / union).item()

def compute_metrics(pred_occ, gt_occ, thr=0.5, eps=1e-6):
    """
    pred_occ: torch.Tensor, shape [1,1,D,H,W], floats in [0,1]
    gt_occ:   torch.Tensor, same shape, floats (0/1 or occupancy scores)
    """
    pred_bin = (pred_occ >= thr)
    gt_bin   = (gt_occ   >= thr)

    tp = (pred_bin & gt_bin).sum().float()
    fp = (pred_bin & ~gt_bin).sum().float()
    fn = (~pred_bin & gt_bin).sum().float()

    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    iou       = tp / (tp + fp + fn + eps)
    f1        = 2 * (precision * recall) / (precision + recall + eps)

    return {
        "precision": precision.item(),
        "recall": recall.item(),
        "iou": iou.item(),
        "f1": f1.item(),
        "tp": tp.item(),
        "fp": fp.item(),
        "fn": fn.item()
    }

def occupied_voxel_centers(grid_xyz, occ, thr=0.5):
    """
    grid_xyz : [3,D,H,W] real-world coordinates of voxel centers
    occ      : [1,1,D,H,W] occupancy in [0,1]
    thr      : threshold to select occupied voxels
    returns  : [N,3] tensor of xyz of occupied voxels (float32, device=grid_xyz.device)
    """
    mask = (occ >= thr).squeeze(0).squeeze(0)  # [D,H,W] bool
    print(f'mask.sum: {mask.sum()}')
    if mask.sum() == 0:
        return torch.empty(0, 3, device=grid_xyz.device, dtype=grid_xyz.dtype)
    x = grid_xyz[0][mask]
    y = grid_xyz[1][mask]
    z = grid_xyz[2][mask]
    return torch.stack([x, y, z], dim=1)

@torch.no_grad()
def precision_recall_radius(pred_pts, gt_pts, tau=0.5, chunk=50000, eps=1e-6):
    """
    pred_pts : [Np,3] predicted points (e.g., occupied voxel centers)
    gt_pts   : [Ng,3] ground-truth points (e.g., occupied voxel centers from LiDAR)
    tau      : radius threshold in meters
    chunk    : chunk size to bound memory for cdist
    returns  : dict with precision, recall, f1, matched_pred, matched_gt, Np, Ng
    """
    device = pred_pts.device if pred_pts.numel() else (gt_pts.device if gt_pts.numel() else "cpu")
    # tau2 = tau * tau

    Np = pred_pts.shape[0]
    Ng = gt_pts.shape[0]

    if Np == 0 and Ng == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "matched_pred": 0, "matched_gt": 0, "Np": 0, "Ng": 0}
    if Np == 0:
        return {"precision": 1.0, "recall": 0.0, "f1": 0.0, "matched_pred": 0, "matched_gt": 0, "Np": 0, "Ng": Ng}
    if Ng == 0:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0, "matched_pred": 0, "matched_gt": 0, "Np": Np, "Ng": 0}

    # --- Precision: for each pred, nearest gt within tau?
    matched_pred = 0
    for s in range(0, Np, chunk):
        e = min(s + chunk, Np)
        d2 = torch.cdist(pred_pts[s:e], gt_pts, p=2)  # [m, Ng]
        # compare squared distances for speed if you want: (pred[:,None,:]-gt[None,:,:]).pow(2).sum(-1)
        min_d2 = d2.min(dim=1)
        min_d2 = min_d2.values
        matched_pred += (min_d2 <= tau).sum().item()

    # --- Recall: for each gt, nearest pred within tau?
    matched_gt = 0
    for s in range(0, Ng, chunk):
        e = min(s + chunk, Ng)
        d2 = torch.cdist(gt_pts[s:e], pred_pts, p=2)  # [m, Np]
        min_d2 = d2.min(dim=1).values
        matched_gt += (min_d2 <= tau).sum().item()

    print(f'matched_pred:{matched_pred}, matched_gt:{matched_gt}')
    precision = matched_pred / (Np + eps)
    recall    = matched_gt / (Ng + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "matched_pred": int(matched_pred),
        "matched_gt": int(matched_gt),
        "Np": int(Np),
        "Ng": int(Ng),
    }

def metrics_radar_splat_style(pred_occ, gt_occ, grid_xyz, thr_pred=0.5, thr_gt=0.5, tau=0.2, chunk=50000):
    """
    Convenience wrapper: get occupied voxel centers then compute distance-based P/R.
    pred_occ, gt_occ : [1,1,D,H,W] tensors in [0,1]
    grid_xyz         : [3,D,H,W] tensor of voxel center coords
    tau              : distance threshold in meters
    """
    pred_pts = occupied_voxel_centers(grid_xyz, pred_occ, thr=0.01)
    gt_pts   = occupied_voxel_centers(grid_xyz, gt_occ, thr=0.01)
    return precision_recall_radius(pred_pts, gt_pts, tau=0.2, chunk=chunk)

# =====================
# GS Model
# =====================
class RadarGaussianField(torch.nn.Module):
    def __init__(self, radar_xyz, radar_power=None, init_sigma=3, device="cpu"):
        super().__init__()
        N = radar_xyz.shape[0]
        self.register_buffer("base_xyz", torch.as_tensor(radar_xyz, device=device, dtype=torch.float32))
        # self.delta = torch.nn.Parameter(torch.zeros(N, 3, device=device))
        self.delta = torch.nn.Parameter(radar_xyz)
        self.log_sigma = torch.nn.Parameter(torch.full((N, 3), math.log(init_sigma), device=device))
        self.logit_opacity = torch.nn.Parameter(torch.full((N, 1), math.log(0.5), device=device))
        if radar_power is not None:
            p = torch.as_tensor(radar_power, device=device, dtype=torch.float32).view(-1,1)
            p = (p - p.mean()) / (p.std() + 1e-6)
            with torch.no_grad():
                self.logit_opacity.copy_(p.clamp(-2, 2))

    def forward(self):
        mu = self.delta #+self.base_xyz
        sigma = torch.exp(self.log_sigma).clamp(1, 29)
        opacity = torch.sigmoid(self.logit_opacity).clamp(1e-5, 0.999)
        return mu, sigma, opacity


def render_occupancy_voxels_local(
    mu, sigma, opacity, grid_xyz,
    cutoff_k=3.0,
    max_splats_per_chunk=1024,
    use_autocast=False
):
    """
    Memory-efficient renderer:
      - Updates only local voxel windows around each splat
      - Chunks splats so intermediates are small
      - Composes occupancy in log-space
    Inputs:
      mu:      (N,3)
      sigma:   (N,3)  (axis-aligned std)
      opacity: (N,1)  (alpha scale)
      grid_xyz: [3, D, H, W] world coords per voxel center
    Returns:
      occ: (1,1,D,H,W) in [0,1]
    """

    device = grid_xyz.device
    D, H, W = grid_xyz.shape[1: ]
    occ = torch.zeros((1,1,D,H,W), device=device)
    # We'll actually maintain log(1-Occ) for numerical stability:
    log1m_occ = torch.zeros_like(occ)  # log(1) = 0

    gx = grid_xyz[0]  # (D,H,W)
    gy = grid_xyz[1]
    gz = grid_xyz[2]

    N = mu.shape[0]
    rng = torch.arange

    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if (use_autocast and torch.cuda.is_available()) else torch.cuda.amp.autocast(enabled=False)
    with autocast_ctx:
        for s in range(0, N, max_splats_per_chunk):
            e = min(s + max_splats_per_chunk, N)
            mu_s = mu[s:e]           # (M,3)
            sig_s = sigma[s:e]       # (M,3)
            opa_s = opacity[s:e, 0]  # (M,)

            # Process each splat independently in a tiny local window
            for j in range(mu_s.shape[0]):
                mx, my, mz = mu_s[j]
                sx, sy, sz = torch.clamp(sig_s[j], 1e-6, 2.0)
                o = torch.clamp(opa_s[j], 1e-6, 0.999)

                # Determine voxel window indices where |dx|<k, |dy|<k, |dz|<k
                # Convert world coords to boolean masks by comparing in normalized space.
                # We'll find index bounds from the continuous inequalities.

                # Compute continuous bounds in world units:
                x_min, x_max = (mx - cutoff_k*sx).item(), (mx + cutoff_k*sx).item()
                y_min, y_max = (my - cutoff_k*sy).item(), (my + cutoff_k*sy).item()
                z_min, z_max = (mz - cutoff_k*sz).item(), (mz + cutoff_k*sz).item()

                # Find index ranges by binary searching precomputed coord arrays:
                # Since gx,gy,gz are 3D grids, we can slice with indices along each axis only
                # if they are separable along axes. In our grid, gx varies along W & H & D;
                # To keep it simple & robust, we'll build index ranges by scanning along each axis:
                # We can retrieve axis vectors by taking a line:
                xv = gx[0,0,:]   # size W
                yv = gy[0,:,0]   # size H
                zv = gz[:,0,0]   # size D

                # Locate index bounds by searching sorted vectors:
                # (Assumes grid is monotonic along axes; make_grid() guarantees that.)
                wx0 = int(torch.searchsorted(xv, torch.tensor(x_min, device=device)).clamp(0, W-1))
                wx1 = int(torch.searchsorted(xv, torch.tensor(x_max, device=device), right=True).clamp(0, W) - 1)
                hy0 = int(torch.searchsorted(yv, torch.tensor(y_min, device=device)).clamp(0, H-1))
                hy1 = int(torch.searchsorted(yv, torch.tensor(y_max, device=device), right=True).clamp(0, H) - 1)
                dz0 = int(torch.searchsorted(zv, torch.tensor(z_min, device=device)).clamp(0, D-1))
                dz1 = int(torch.searchsorted(zv, torch.tensor(z_max, device=device), right=True).clamp(0, D) - 1)

                if (wx1 < wx0) or (hy1 < hy0) or (dz1 < dz0):
                    continue  # window out of bounds or empty

                # Local voxel coords
                gx_loc = gx[dz0:dz1+1, hy0:hy1+1, wx0:wx1+1]
                gy_loc = gy[dz0:dz1+1, hy0:hy1+1, wx0:wx1+1]
                gz_loc = gz[dz0:dz1+1, hy0:hy1+1, wx0:wx1+1]

                dx = (gx_loc - mx) / sx
                dy = (gy_loc - my) / sy
                dz = (gz_loc - mz) / sz

                g = torch.exp(-0.5*(dx*dx + dy*dy + dz*dz))  # (dz, hy, wx)

                # alpha per voxel for this splat:
                # α = 1 - exp(-o * g); keep it small and stable
                alpha = 1.0 - torch.exp(-o * g)

                # log-space composition: log(1-Occ_new) = log( (1-Occ_old) * (1-α) )
                # => log1m_occ_local += log(1 - alpha)
                log1m_update = torch.log1p(-alpha.clamp(0, 0.999999))

                # In-place update of the window
                log1m_occ[:, :, dz0:dz1+1, hy0:hy1+1, wx0:wx1+1] += log1m_update.unsqueeze(0).unsqueeze(0)

        # Convert back to occupancy: Occ = 1 - exp(log(1-Occ))
        occ = 1.0 - torch.exp(log1m_occ)
        return occ.clamp(0, 1)

def find_timestamp(path: str):
    """Extract numeric frame index, e.g. os2_64_00001.pcd → 1."""
    name = os.path.splitext(os.path.basename(path))[0].split('_')[-1]
    digits = ''.join(ch for ch in name if ch.isdigit())
    return int(digits) if digits else None

from pypcd4 import PointCloud
def safe_load(path: str) -> np.ndarray:
    """
    Supports .npy, .npz (expects 'arr_0' or 'data' key), or simple .txt/.bin with whitespace.
    You can extend this to custom formats if needed.
    """
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".npy":
        return np.load(path)
    if ext == ".npz":
        z = np.load(path)
        for k in ("arr_0","data","points"):
            if k in z: return z[k]
        # fallback: first array
        k = list(z.keys())[0]
        return z[k]
    if ext in (".txt",".csv"):
        return np.loadtxt(path, delimiter=',' if ext=='.csv' else None)
    if ext in (".bin",):
        # If LiDAR .bin is KITTI-style (x,y,z,reflectance) float32:
        try:
            arr = np.fromfile(path, dtype=np.float32).reshape(-1,4)
            return arr
        except Exception:
            # as a last resort, try float32 triplets
            arr = np.fromfile(path, dtype=np.float32)
            c = 3 if arr.size%3==0 else 4
            return arr.reshape(-1, c)
    if ext in ('.pcd'): # (x,y,z,reflectance)
        pc = PointCloud.from_path(path)
        points = pc.numpy()
        return points[:, :4]
    
    raise ValueError(f"Unsupported file format: {path}")

@torch.no_grad()
def nms_localmax_nd(occ: torch.Tensor, radius_vox: int, min_conf: float):
    """
    N-D local-maxima NMS via max-pooling.
    occ: [H,W] or [D,H,W], dtype float
    radius_vox: neighborhood radius in voxels (scalar, use anisotropic if needed in a custom version)
    min_conf: minimum occupancy value to consider a peak
    """
    occ = occ.squeeze()
    dims = occ.dim()
    assert dims in (2,3), "Only BEV [H,W] or 3D [D,H,W] supported."
    x = occ.unsqueeze(0).unsqueeze(0)  # [1,1,...]
    if dims == 2:
        k = 2*radius_vox + 1
        pool = F.max_pool2d(x, kernel_size=k, stride=1, padding=radius_vox)
    else:
        k = 2*radius_vox + 1
        pool = F.max_pool3d(x, kernel_size=k, stride=1, padding=radius_vox)
    peaks = (x == pool) & (x >= min_conf)
    return peaks.squeeze(0).squeeze(0)  # shape [H,W] or [D,H,W], bool

def derive_grid_params_from_grid_xyz(grid_xyz):
    """
    grid_xyz: Tensor shaped [D,H,W,3] for 3D or [H,W,3] for BEV,
              giving world (x,y,z) of each voxel center.
    Returns: voxel_size_m (vx,vy,vz), grid_origin_m (ox,oy,oz)
    """
    import torch
    dims = grid_xyz.dim()
    if dims == 3:  # BEV [H,W,3], z is constant plane
        H, W, _ = grid_xyz.shape
        cx00 = grid_xyz[0,0]         # (x,y,z) of voxel (0,0)
        cx01 = grid_xyz[0,1] if W > 1 else cx00
        cx10 = grid_xyz[1,0] if H > 1 else cx00
        vx = (cx01[0] - cx00[0]).abs().item()
        vy = (cx10[1] - cx00[1]).abs().item()
        vz = 1.0  # BEV slab; set to your chosen slab thickness
        ox, oy, oz = cx00.tolist()
        return (vx, vy, vz), (ox, oy, oz)
    elif dims == 4:  # 3D [D,H,W,3]
        D, H, W, _ = grid_xyz.shape
        c000 = grid_xyz[0,0,0]
        c001 = grid_xyz[0,0,1] if W > 1 else c000
        c010 = grid_xyz[0,1,0] if H > 1 else c000
        c100 = grid_xyz[1,0,0] if D > 1 else c000
        vx = (c001[0] - c000[0]).abs().item()
        vy = (c010[1] - c000[1]).abs().item()
        vz = (c100[2] - c000[2]).abs().item()
        ox, oy, oz = c000.tolist()
        return (vx, vy, vz), (ox, oy, oz)
    else:
        raise ValueError("grid_xyz must be [H,W,3] or [D,H,W,3]")


@torch.no_grad()
def extract_centroids_metric(
    occ: torch.Tensor,
    peaks_mask: torch.Tensor,
    window_radius: int,
    voxel_size_m: tuple,
    grid_origin_m: tuple,
    layout: str = "DHWinZYX",
    score_mode: str = "integral",
):
    """
    For each peak voxel, compute a power-weighted centroid in voxel space,
    then convert to metric (x,y,z). Works for BEV [H,W] or 3D [D,H,W].

    voxel_size_m: (vx, vy, vz) in meters **of the world axes (x forward, y left, z up)**
    grid_origin_m: (ox, oy, oz) metric coord of voxel (z=0,y=0,x=0) corner
                   (for BEV, oz can be the ground plane or lowest z of grid)
    layout: how occ dimensions map to world axes:
        - "HW->YX":         occ[y,x] BEV; z fixed; y first, x second
        - "DHWinZYX":       occ[z,y,x] 3D grid; z first, y second, x last
      (Use these names exactly as below.)
    score_mode: "integral" (sum in window) or "peak" (max in window)
    """
    assert score_mode in ("integral", "peak")
    occ = occ.squeeze()
    dims = occ.dim()
    assert dims in (2,3)

    points = []  # list of (x,y,z,score)
    if dims == 2:
        H, W = occ.shape
        ys, xs = torch.nonzero(peaks_mask, as_tuple=True)
        for y, x in zip(ys.tolist(), xs.tolist()):
            y0, y1 = max(0, y-window_radius), min(H, y+window_radius+1)
            x0, x1 = max(0, x-window_radius), min(W, x+window_radius+1)
            patch = occ[y0:y1, x0:x1]
            if patch.numel() == 0:
                continue
            # power-weighted centroid in voxel coords
            yy, xx = torch.meshgrid(
                torch.arange(y0, y1, device=occ.device),
                torch.arange(x0, x1, device=occ.device),
                indexing='ij'
            )
            w = patch.clamp_min(0)
            mass = w.sum()
            if mass <= 0:
                continue
            cx_vox = (w * xx).sum() / (mass + 1e-12)
            cy_vox = (w * yy).sum() / (mass + 1e-12)
            score = mass if score_mode == "integral" else patch.max()

            # Convert to metric using BEV YX -> (y,x) mapping
            # layout "HW->YX": x_world = ox + cx*vx, y_world = oy + cy*vy, z_world fixed by origin oz
            vx, vy, vz = voxel_size_m
            ox, oy, oz = grid_origin_m
            x_m = ox + cx_vox * vx
            y_m = oy + cy_vox * vy
            z_m = oz  # BEV: single layer (set oz appropriately, e.g., ground plane or mid-slab)
            points.append((x_m.item(), y_m.item(), z_m if isinstance(z_m, float) else float(z_m), float(score)))
        return torch.tensor(points, dtype=torch.float32, device=occ.device) if points else \
               torch.zeros((0,4), dtype=torch.float32, device=occ.device)

    else:
        D, H, W = occ.shape
        zs, ys, xs = torch.nonzero(peaks_mask, as_tuple=True)
        for z, y, x in zip(zs.tolist(), ys.tolist(), xs.tolist()):
            z0, z1 = max(0, z-window_radius), min(D, z+window_radius+1)
            y0, y1 = max(0, y-window_radius), min(H, y+window_radius+1)
            x0, x1 = max(0, x-window_radius), min(W, x+window_radius+1)
            patch = occ[z0:z1, y0:y1, x0:x1]
            if patch.numel() == 0:
                continue
            zz, yy, xx = torch.meshgrid(
                torch.arange(z0, z1, device=occ.device),
                torch.arange(y0, y1, device=occ.device),
                torch.arange(x0, x1, device=occ.device),
                indexing='ij'
            )
            w = patch.clamp_min(0)
            mass = w.sum()
            if mass <= 0:
                continue
            cx_vox = (w * xx).sum() / (mass + 1e-12)
            cy_vox = (w * yy).sum() / (mass + 1e-12)
            cz_vox = (w * zz).sum() / (mass + 1e-12)
            score = mass if score_mode == "integral" else patch.max()

            # Convert voxel indices -> metric (x,y,z)
            # layout "DHWinZYX": occ[z,y,x] corresponds to world (z,y,x)
            vx, vy, vz = voxel_size_m
            ox, oy, oz = grid_origin_m
            x_m = ox + cx_vox * vx
            y_m = oy + cy_vox * vy
            z_m = oz + cz_vox * vz
            points.append((x_m.item(), y_m.item(), z_m.item(), float(score)))
        return torch.tensor(points, dtype=torch.float32, device=occ.device) if points else \
               torch.zeros((0,4), dtype=torch.float32, device=occ.device)


@torch.no_grad()
def occupancy_to_sparse_points(
    occ: torch.Tensor,
    voxel_size_m=(0.5, 0.5, 0.5),
    grid_origin_m=(0.0, 0.0, 0.0),
    nms_radius_vox=2,
    min_conf=0.2,
    window_radius=2,
    layout=None,         # auto: "HW->YX" if 2D, "DHWinZYX" if 3D
    score_mode="integral"
):
    """
    High-level helper: run NMS + centroid extraction and get [N,4] points (x,y,z,score).
    """
    if layout is None:
        layout = "HW->YX" if occ.dim() == 2 else "DHWinZYX"
    peaks = nms_localmax_nd(occ, radius_vox=nms_radius_vox, min_conf=min_conf)
    peaks = True*torch.ones_like(peaks).to(peaks.device)
    pts = extract_centroids_metric(
        occ, peaks, window_radius=window_radius,
        voxel_size_m=voxel_size_m, grid_origin_m=grid_origin_m,
        layout=layout, score_mode=score_mode
    )
    return pts  # [N,4] float32 on same device


@torch.no_grad()
def to_kradar_sparse_input(points_xyzs: torch.Tensor, want_fields="xyzp"):
    """
    Map (x,y,z,score) -> a K-Radar-friendly sparse tensor.
    want_fields: string describing desired order, e.g. "xyzp", "xyzpv" (add vx,vy zeros), etc.
    Returns tensor [N,C] in the requested order.
    """
    assert points_xyzs.shape[-1] == 4, "Expect [N,4]=(x,y,z,score)"
    x, y, z, p = [points_xyzs[:, i:i+1] for i in range(4)]
    cols = []
    for ch in want_fields:
        if ch == 'x': cols.append(x)
        elif ch == 'y': cols.append(y)
        elif ch == 'z': cols.append(z)
        elif ch == 'p': cols.append(p)           # score/power/prob as 'p'
        elif ch == 'r': cols.append(torch.ones_like(x))  # reflectivity placeholder
        elif ch == 'd': cols.append(torch.zeros_like(x)) # doppler placeholder
        elif ch == 'u': cols.append(torch.zeros_like(x)) # vx placeholder
        elif ch == 'v': cols.append(torch.zeros_like(x)) # vy placeholder
        else:
            raise ValueError(f"Unsupported channel code '{ch}'")
    return torch.cat(cols, dim=1).contiguous()
