
# (content truncated in the last run) — full module is rewritten here
from typing import Tuple, Optional
import numpy as np

try:
    import torch
except Exception:
    torch = None

try:
    import open3d as o3d
    _HAS_O3D = True
except Exception:
    o3d = None
    _HAS_O3D = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


def to_numpy(x, shape_last: Optional[int]=None, dtype=np.float64):
    if x is None:
        return None
    if torch is not None and isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=dtype, order="C")
    if shape_last is not None and (x.ndim < 2 or x.shape[-1] != shape_last):
        raise ValueError(f"Expected last dim={shape_last}, got {x.shape}")
    return x


def voxel_indices_to_world_xyz(indices, voxel_size: Tuple[float,float,float], origin: Tuple[float,float,float]):
    if torch is not None and isinstance(indices, torch.Tensor):
        idx = indices.long()
        z = idx[:,1].to(torch.float32); y = idx[:,2].to(torch.float32); x = idx[:,3].to(torch.float32)
        vx, vy, vz = voxel_size; ox, oy, oz = origin
        xc = ox + (x + 0.5) * vx
        yc = oy + (y + 0.5) * vy
        zc = oz + (z + 0.5) * vz
        out = torch.stack([xc, yc, zc], dim=1)
        return to_numpy(out, shape_last=3, dtype=np.float64)
    else:
        idx = np.asarray(indices, dtype=np.int64)
        z = idx[:,1].astype(np.float64); y = idx[:,2].astype(np.float64); x = idx[:,3].astype(np.float64)
        vx, vy, vz = voxel_size; ox, oy, oz = origin
        xc = ox + (x + 0.5) * vx
        yc = oy + (y + 0.5) * vy
        zc = oz + (z + 0.5) * vz
        return np.stack([xc, yc, zc], axis=1)


def predicted_slots_to_xyz(pred_indices, pred_offsets_vox, voxel_size: Tuple[float,float,float], origin: Tuple[float,float,float]):
    centers = voxel_indices_to_world_xyz(pred_indices, voxel_size, origin)
    off = to_numpy(pred_offsets_vox, shape_last=3, dtype=np.float64)
    vx, vy, vz = voxel_size
    scale = np.array([vx, vy, vz], dtype=np.float64).reshape(1,1,3)
    xyz = centers.reshape(-1,1,3) + off * scale
    return xyz.reshape(-1,3)


def apply_rigid(X, R=None, t=None):
    X = to_numpy(X, shape_last=3, dtype=np.float64)
    if R is not None:
        R = to_numpy(R, dtype=np.float64)
        X = (R @ X.T).T
    if t is not None:
        t = to_numpy(t, dtype=np.float64).reshape(1,3)
        X = X + t
    return X


def compute_reference_pose(X_ref, view="bev", margin_scale=1.5):
    X = to_numpy(X_ref, shape_last=3, dtype=np.float64)
    if X.size == 0:
        ctr = np.array([0.0,0.0,0.0], dtype=np.float64)
        extent = 10.0
    else:
        a = X.min(axis=0); b = X.max(axis=0)
        ctr = (a + b) / 2.0
        extent = float(np.max(b - a))
        if extent <= 1e-6: extent = 10.0
    if view == "bev":
        eye = ctr + np.array([0.001, 0.001, margin_scale*extent], dtype=np.float64)
        up  = np.array([0,1,0], dtype=np.float64)
    elif view == "front":
        eye = ctr + np.array([margin_scale*extent, 0.0, 0.001], dtype=np.float64)
        up  = np.array([0,0,1], dtype=np.float64)
    else:
        eye = ctr + np.array([0.001, margin_scale*extent, 0.001], dtype=np.float64)
        up  = np.array([0,0,1], dtype=np.float64)
    return {"ctr": ctr, "eye": eye, "up": up}


def _colors_from_intensity(I, N):
    if I is None:
        C = np.tile(np.array([[0.2,0.7,1.0]], dtype=np.float64), (N,1))
    else:
        I = to_numpy(I, dtype=np.float64).reshape(-1)
        if I.size != N:
            if I.size == 1:
                I = np.full((N,), I.item(), dtype=np.float64)
            else:
                I = np.resize(I, (N,))
        v = (I - np.min(I)) / (np.ptp(I) + 1e-6)
        C = np.stack([v, np.minimum(1.0, 0.5 + 0.5*v), 1.0 - v], axis=1).astype(np.float64)
    return C


def save_open3d_render_fixed_pose(points_xyz, intensities=None, filename="view.png",
                                  pose=None, width=1600, height=900,
                                  fov_deg=60.0, near=0.1, far=5000.0,
                                  bg=(1,1,1,1), point_size=1.5):
    X = to_numpy(points_xyz, shape_last=3, dtype=np.float64)
    N = X.shape[0]
    C = _colors_from_intensity(intensities, N)

    if _HAS_O3D:
        rnd = o3d.visualization.rendering.OffscreenRenderer(width, height)
        rnd.scene.set_background(bg)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(X.copy()))
        pcd.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(C.copy()))

        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = float(point_size)

        rnd.scene.add_geometry("points", pcd, mat)

        if pose is None:
            pose = compute_reference_pose(X, view="bev")
        cam = rnd.scene.camera
        cam.look_at(pose["ctr"], pose["eye"], pose["up"])
        cam.set_projection(
            float(fov_deg),
            float(width)/float(height),
            float(near),
            float(far),
            o3d.visualization.rendering.Camera.FovType.Vertical
        )

        img = rnd.render_to_image()
        o3d.io.write_image(filename, img)
        return filename

    # Fallback to matplotlib BEV
    if not _HAS_MPL:
        raise RuntimeError("Neither Open3D nor matplotlib available for rendering.")
    x, y = X[:,0], X[:,1]
    xr = (x.min(), x.max()); yr = (y.min(), y.max())
    W = 1200; H = 800
    fig, ax = plt.subplots(figsize=(W/100, H/100), dpi=100)
    ax.set_facecolor((1,1,1))
    ax.scatter(x, y, s=1, c=C, marker=".")
    ax.set_aspect("equal")
    ax.set_xlim(*xr); ax.set_ylim(*yr)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(filename, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return filename


def save_triplet_views(pred_xyz, radar_xyz, lidar_xyz,
                       pred_I=None, radar_I=None, lidar_I=None,
                       pose=None, out_dir=".", prefix="frame",
                       width=1600, height=900):
    import os
    os.makedirs(out_dir, exist_ok=True)
    if pose is None:
        pose = compute_reference_pose(lidar_xyz if lidar_xyz is not None else pred_xyz, view="bev")
    paths = {}
    paths["pred"] = save_open3d_render_fixed_pose(pred_xyz, pred_I, os.path.join(out_dir, f"{prefix}_pred.png"), pose, width, height)
    if radar_xyz is not None:
        paths["rdr"]  = save_open3d_render_fixed_pose(radar_xyz, radar_I, os.path.join(out_dir, f"{prefix}_rdr.png"),  pose, width, height)
    if lidar_xyz is not None:
        paths["ldr"]  = save_open3d_render_fixed_pose(lidar_xyz, lidar_I, os.path.join(out_dir, f"{prefix}_ldr.png"),  pose, width, height)
    return paths

import torch
import numpy as np

def unet_slots_to_xyz_attrs(pred, voxel_size, origin, prob_thresh=0.3, clamp_offsets=True):
    """
    pred: {"st": SparseConvTensor, "logits": [N,K], "offs": [N,K,3], "attrs": [N,K,F]}
    voxel_size: (vx, vy, vz) meters
    origin:     (ox, oy, oz) meters
    Returns:
      xyz  : (M,3) float64 world coords (meters)
      attrs: (M,F) float64 attributes aligned with xyz (e.g., intensity in [:,0])
    """
    st      = pred["st"]
    logits  = pred["logits"]     # [N,K]
    offs    = pred["offs"]       # [N,K,3]  (voxel units!)
    attrs   = pred["attrs"]      # [N,K,F]

    # 1) centers from spconv indices [b,z,y,x]  →  [x,y,z] meters
    idx = st.indices.long()      # [N,4]
    z = idx[:,1].float(); y = idx[:,2].float(); x = idx[:,3].float()
    vx, vy, vz = voxel_size
    ox, oy, oz = origin
    centers = torch.stack([
        ox + (x + 0.5) * vx,
        oy + (y + 0.5) * vy,
        oz + (z + 0.5) * vz
    ], dim=1)                     # [N,3] meters

    # 2) select valid slots by probability
    probs = torch.sigmoid(logits)                 # [N,K]
    keep  = probs >= prob_thresh                  # [N,K] bool

    # 3) offsets: voxel units → meters
    if clamp_offsets:
        offs = torch.clamp(offs, -0.5, 0.5)       # optional, keeps points inside voxel
    scale = offs.new_tensor([vx, vy, vz])         # [3]
    offs_m = offs * scale                         # [N,K,3] meters

    # 4) assemble world xyz per slot
    xyz = centers[:,None,:] + offs_m              # [N,K,3]
    F   = attrs.shape[-1]
    xyz  = xyz[keep].detach().cpu().numpy().astype(np.float64)    # (M,3)
    attrs= attrs[keep].detach().cpu().numpy().astype(np.float64)  # (M,F)
    return xyz, attrs
