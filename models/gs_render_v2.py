# differentiable_renderer.py
import torch
import torch.nn.functional as F
from contextlib import nullcontext

def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """
    q: (N,4) quaternion (w,x,y,z). Returns (N,3,3) rotation matrices.
    """
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    R = torch.stack([
        ww+xx-yy-zz, 2*(xy-wz),     2*(xz+wy),
        2*(xy+wz),   ww-xx+yy-zz,   2*(yz-wx),
        2*(xz-wy),   2*(yz+wx),     ww-xx-yy+zz
    ], dim=-1).reshape(q.shape[:-1] + (3,3))
    return R

@torch.no_grad()
def _grid_metrics_from_xyz(grid_xyz: torch.Tensor):
    """
    grid_xyz: (3, D, H, W) regular grid in meters.
    Returns voxel sizes (vx,vy,vz) and origin (ox,oy,oz).
    """
    _, D, H, W = grid_xyz.shape
    gx, gy, gz = grid_xyz[0], grid_xyz[1], grid_xyz[2]

    def _delta_along(arr: torch.Tensor, axis: int) -> float:
        # robust central diffs; assumes regular spacing
        if axis == 2:   # x (W)
            diffs = arr[..., 2:] - arr[..., 1:-1]
        elif axis == 1: # y (H)
            diffs = arr[:, 2:, :] - arr[:, 1:-1, :]
        else:           # z (D)
            diffs = arr[2:, ...] - arr[1:-1, ...]
        # if grid is exactly regular, these are constant; take mean abs
        return diffs.nanmean().abs().item()

    vx = max(_delta_along(gx, 2), 1e-12)
    vy = max(_delta_along(gy, 1), 1e-12)
    vz = max(_delta_along(gz, 0), 1e-12)
    ox = gx[0, 0, 0].item()
    oy = gy[0, 0, 0].item()
    oz = gz[0, 0, 0].item()
    return vx, vy, vz, ox, oy, oz

def render_occupancy_voxels_local_diff(
    mu: torch.Tensor,                 # (N,3) meters
    sigma: torch.Tensor,              # (N,) or (N,3) meters
    opacity: torch.Tensor,            # (N,) >= 0  (rate scale for Poisson; alpha scale for "alpha")
    grid_xyz: torch.Tensor,           # (3, D, H, W) meters (regular grid)
    cutoff_k: float = 3.0,            # evaluate within +-k std (perf window)
    max_splats_per_chunk: int = 4096, # memory/perf control
    use_autocast: bool = True,        # match old flag; autocast on CUDA/MPS
    quat: torch.Tensor = None,        # (N,4) (w,x,y,z) orientation; None = identity
    compositor: str = "poisson",      # "poisson" (O=1-e^{-S}) or "alpha" (order-indep alpha blend)
) -> torch.Tensor:
    """
    Fully differentiable per-splat Gaussian renderer (anisotropic, oriented).
    Returns:
        pred_occ: (D, H, W) occupancy in [0,1]
    Differentiable w.r.t. mu, sigma, opacity, quat (inside the AABB window).
    """
    assert grid_xyz.dim() == 4 and grid_xyz.shape[0] == 3, "grid_xyz must be (3,D,H,W)"
    device = mu.device
    _, D, H, W = grid_xyz.shape

    # infer voxel metrics from the grid you already use
    vx, vy, vz, ox, oy, oz = _grid_metrics_from_xyz(grid_xyz)

    N = mu.shape[0]
    if sigma.dim() == 1:
        sigma = sigma.unsqueeze(-1).expand(-1, 3)
    assert sigma.shape == (N, 3), "sigma must be (N,) or (N,3) in METERS"

    if quat is None:
        quat = torch.zeros(N, 4, device=device)
        quat[:, 0] = 1.0  # identity

    # rotations & inverse covariances per splat
    R = _quat_to_rotmat(quat)                             # (N,3,3)
    sig2 = (sigma.clamp_min(1e-6)) ** 2                   # (N,3)
    invSigLocal = torch.zeros(N, 3, 3, device=device)
    invSigLocal[:, 0, 0] = 1.0 / sig2[:, 0]
    invSigLocal[:, 1, 1] = 1.0 / sig2[:, 1]
    invSigLocal[:, 2, 2] = 1.0 / sig2[:, 2]
    invSigma = R @ invSigLocal @ R.transpose(-1, -2)      # (N,3,3)

    # world-axis stds only for bounds (perf only, not in math)
    cov_world = R @ torch.diag_embed(sig2) @ R.transpose(-1, -2)  # (N,3,3)
    std_world = torch.sqrt(torch.stack(
        [cov_world[:, 0, 0], cov_world[:, 1, 1], cov_world[:, 2, 2]], dim=-1))  # (N,3)
    std_vox = torch.stack([std_world[:, 0] / vx,
                           std_world[:, 1] / vy,
                           std_world[:, 2] / vz], dim=-1)  # (N,3) in voxel units

    # allocate accumulator
    if compositor == "poisson":
        acc = torch.zeros(D, H, W, device=device)  # rate S
    elif compositor == "alpha":
        acc = torch.zeros(D, H, W, device=device)  # sum log(1 - aG)
    else:
        raise ValueError("compositor must be 'poisson' or 'alpha'")

    # centers in index space (float)
    cx = (mu[:, 0] - ox) / vx
    cy = (mu[:, 1] - oy) / vy
    cz = (mu[:, 2] - oz) / vz

    # per-splat radius (in voxels)
    rx = (cutoff_k * std_vox[:, 0]).clamp_min(1.0)
    ry = (cutoff_k * std_vox[:, 1]).clamp_min(1.0)
    rz = (cutoff_k * std_vox[:, 2]).clamp_min(1.0)

    # integer windows (compute mask only; gradients flow through values inside)
    x0 = (cx - rx).floor().clamp(0, W - 1).long()
    x1 = (cx + rx).ceil().clamp(0, W - 1).long() + 1
    y0 = (cy - ry).floor().clamp(0, H - 1).long()
    y1 = (cy + ry).ceil().clamp(0, H - 1).long() + 1
    z0 = (cz - rz).floor().clamp(0, D - 1).long()
    z1 = (cz + rz).ceil().clamp(0, D - 1).long() + 1

    dt = str(device).split(':')[0]
    amp_ctx = (torch.autocast(device_type=dt, dtype=torch.float16)
               if use_autocast and device.type in ("cuda", "mps") else nullcontext())

    with amp_ctx:
        for i0 in range(0, N, max_splats_per_chunk):
            i1 = min(N, i0 + max_splats_per_chunk)
            for i in range(i0, i1):
                if (x1[i] <= x0[i]) or (y1[i] <= y0[i]) or (z1[i] <= z0[i]) or (opacity[i] <= 0):
                    continue

                # local voxel index grid
                zz = torch.arange(z0[i], z1[i], device=device, dtype=torch.float32)
                yy = torch.arange(y0[i], y1[i], device=device, dtype=torch.float32)
                xx = torch.arange(x0[i], x1[i], device=device, dtype=torch.float32)
                Z, Y, X = torch.meshgrid(zz, yy, xx, indexing='ij')

                # metric coords via grid metrics (regular grid assumed)
                x_m = ox + X * vx
                y_m = oy + Y * vy
                z_m = oz + Z * vz

                # per-voxel offsets to center (meters)
                d = torch.stack([x_m - mu[i, 0],
                                 y_m - mu[i, 1],
                                 z_m - mu[i, 2]], dim=-1)  # (...,3)

                invS = invSigma[i]  # (3,3)
                tmp = torch.matmul(d, invS.transpose(0, 1))   # (...,3)
                m2 = (tmp * d).sum(dim=-1).clamp_min(0.0)     # Mahalanobis^2
                G = torch.exp(-0.5 * m2)                      # unnormalized Gaussian
                a = opacity[i].clamp_min(0.0).to(device)

                if compositor == "poisson":
                    acc[z0[i]:z1[i], y0[i]:y1[i], x0[i]:x1[i]] += a * G
                else:
                    aG = (a * G).clamp(0, 0.999999)
                    acc[z0[i]:z1[i], y0[i]:y1[i], x0[i]:x1[i]] += torch.log1p(-aG)

    if compositor == "poisson":
        pred_occ = 1.0 - torch.exp(-acc.clamp_min(0))
    else:
        pred_occ = 1.0 - torch.exp(acc)  # acc holds sum log(1 - aG)

    return pred_occ