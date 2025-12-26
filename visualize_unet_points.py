
# (content truncated in the last run) — full module is rewritten here
from typing import Tuple, Optional
import numpy as np
from scipy.spatial import cKDTree

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

def make_o3d_box(x, y, z, th, l, w, h, color=[1, 0, 0]):
    """
    Create an Open3D oriented box from K-Radar format:
    (x, y, z, yaw, length, width, height)
    """
    box = o3d.geometry.OrientedBoundingBox()

    # Center
    box.center = np.array([x, y, z])

    # Rotation around Z
    R = box.get_rotation_matrix_from_xyz([0, 0, th])
    box.R = R

    # Extents (length, width, height)
    box.extent = np.array([l, w, h])

    box.color = np.array(color)
    return box

def save_open3d_render_fixed_pose(points_xyz, intensities=None, boxes=None, filename="view.png",
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

        # ---- Bounding Boxes ----
        for i, box_vals in enumerate(boxes):
            cls_name, (x, y, z, th, l, w, h), trk, avail = box_vals
            box_obj = make_o3d_box(x, y, z, th, l, w, h)

            mat_box = o3d.visualization.rendering.MaterialRecord()
            mat_box.shader = "unlitLine"
            mat_box.line_width = 3.0

            rnd.scene.add_geometry(f"box_{i}", box_obj, mat_box)

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

import matplotlib.cm as cm
import matplotlib.colors as mcolors
def apply_colormap(values, cmap_name="viridis", vmin=None, vmax=None):
    values = np.asarray(values)
    if vmin is None: vmin = np.nanmin(values)
    if vmax is None: vmax = np.nanmax(values)

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = cm.get_cmap(cmap_name)

    colors = cmap(norm(values))[:, :3]  # drop alpha
    return colors

def save_open3d_render_offsets(points_xyz, points_gt, points_rdr, intensities, intensities_gt, intensities_rdr, boxes=None, filename="view.png",
                                  pose=None, width=1600, height=900,
                                  fov_deg=60.0, near=0.1, far=5000.0,
                                  bg=(1,1,1,1), point_size=1.5):
    X = to_numpy(points_xyz, shape_last=3, dtype=np.float64) # [N, 3]
    N = X.shape[0]
    C = _colors_from_intensity(intensities, N) # [N, ]

    Xgt = to_numpy(points_gt, shape_last=3, dtype=np.float64) # [N, 3]
    Cgt = _colors_from_intensity(intensities_gt, N) # [N, ]

    Xrdr = to_numpy(points_rdr, shape_last=3, dtype=np.float64) # [N, 3]
    Crdr = _colors_from_intensity(intensities_rdr, N) # [N, ]

    if _HAS_O3D:
        rnd = o3d.visualization.rendering.OffscreenRenderer(width, height)
        rnd.scene.set_background(bg)

        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(np.ascontiguousarray(Xgt.copy()))
        pcd_gt.colors = o3d.utility.Vector3dVector(apply_colormap(intensities_gt, cmap_name="viridis"))

        pcd_pred = o3d.geometry.PointCloud()
        pcd_pred.points = o3d.utility.Vector3dVector(np.ascontiguousarray(X.copy()))
        pcd_pred.colors = o3d.utility.Vector3dVector(apply_colormap(intensities, cmap_name="plasma"))
                                                     
        pcd_rdr = o3d.geometry.PointCloud()
        pcd_rdr.points = o3d.utility.Vector3dVector(np.ascontiguousarray(Xrdr.copy()))
        pcd_rdr.colors = o3d.utility.Vector3dVector(apply_colormap(intensities_rdr, cmap_name="coolwarm"))

        mat_gt = o3d.visualization.rendering.MaterialRecord()
        mat_gt.shader = "defaultUnlit"
        mat_gt.point_size = 1.5

        mat_pred = o3d.visualization.rendering.MaterialRecord()
        mat_pred.shader = "defaultUnlit"
        mat_pred.point_size = 2.5

        mat_rdr = o3d.visualization.rendering.MaterialRecord()
        mat_rdr.shader = "defaultUnlit"
        mat_rdr.point_size = 3.5

        rnd.scene.add_geometry("gt", pcd_gt, mat_gt)
        rnd.scene.add_geometry("pred", pcd_pred, mat_pred)
        rnd.scene.add_geometry("rdr", pcd_rdr, mat_rdr)

        # ---- Bounding Boxes ----
        for i, box_vals in enumerate(boxes):
            cls_name, (x, y, z, th, l, w, h), trk, avail = box_vals
            box_obj = make_o3d_box(x, y, z, th, l, w, h)

            mat_box = o3d.visualization.rendering.MaterialRecord()
            mat_box.shader = "unlitLine"
            mat_box.line_width = 3.0

            rnd.scene.add_geometry(f"box_{i}", box_obj, mat_box)

        if pose is None:
            pose = compute_reference_pose(Xgt, view="bev")
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
    # ax.scatter(x, y, s=1, c=C, marker=".", cmap="viridis")

    xgt, ygt = Xgt[:,0], Xgt[:,1]
    ax.scatter(xgt, ygt, s=1, c=Cgt, marker=".", cmap="coolwarm")

    ax.set_aspect("equal")
    ax.set_xlim(*xr); ax.set_ylim(*yr)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(filename, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return filename

def plot_quiver(pts_pred, off_pred, name=''):

    # BEV plane
    x = pts_pred[:, 0]   # meters
    y = pts_pred[:, 1]   # meters

    dx_gt  = off_pred[:, 0]
    dy_gt  = off_pred[:, 1]

    # downsample
    step = 10
    idx = np.arange(0, len(x), step)

    plt.figure(figsize=(6, 6))

    # background points
    # plt.scatter(x[idx], y[idx], s=3, c="gray", alpha=0.3)

    # arrow scale (meters → display)
    arrow_scale = 10.0   # ↑ larger → shorter arrows

    # pred → gt
    plt.quiver(
        x[idx], y[idx],
        dx_gt[idx], dy_gt[idx],
        angles="xy",
        scale_units="xy",
        scale=arrow_scale,
        color="blue",
        width=0.002,
        label="pred → gt"
    )


    plt.axis("equal")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Offset vectors in BEV (meters)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(name)


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

def unet_slots_to_xyz_attrs(pred, offs, occ, voxel_size, origin, prob_thresh=0.3, clamp_offsets=True):
    """
    pred: {"st": SparseConvTensor, "logits": [N,K], "offs": [N,K,3], "attrs": [N,K,F]}
    voxel_size: (vx, vy, vz) meters
    origin:     (ox, oy, oz) meters
    Returns:
      xyz  : (M,3) float64 world coords (meters)
      attrs: (M,F) float64 attributes aligned with xyz (e.g., intensity in [:,0])
    """
    st      = pred["st"]
    # logits  = pred["logits"]     # [N,K]
    # offs    = pred["offs"]       # [N,K,3]  (voxel units!)
    # attrs   = pred["attrs"]      # [N,K,F]
    # offs    = attrs[:, :, :3]    # [N,K,3]  (voxel units!)


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
    # probs = torch.sigmoid(logits)                 # [N,K]
    keep  = occ >= prob_thresh                  # [N,K, 1] bool
    print(f'keep shape:{keep.shape}')
    keep = keep.squeeze()

    # 3) offsets: voxel units → meters
    if clamp_offsets:
        offs = torch.clamp(offs, -0.5, 0.5)       # optional, keeps points inside voxel
    scale = offs.new_tensor([vx, vy, vz])         # [3]
    offs_m = offs * scale                         # [N,K,3] meters
    print('centers: ', centers)

    # 4) assemble world xyz per slot
    xyz = centers[:, :].unsqueeze(1).repeat(1, 5, 1) + offs_m              # [N,K,3]
    # F   = attrs.shape[-1]
    xyz  = xyz[keep].detach().cpu().numpy().astype(np.float64)    # (M,3)
    # attrs= attrs[keep].detach().cpu().numpy().astype(np.float64)  # (M,F)
    return xyz

def nn_error_vs_x_numpy_with_zero(gt_points, pred_points, num_bins, x_min, x_max):
    """
    Compute NN error per x-bin (in meters), returning 0 where no GT exists.
    """
    bins = np.linspace(x_min, x_max, num_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    mean_errors = np.zeros(num_bins, dtype=np.float64)
    counts = np.zeros(num_bins, dtype=np.int32)

    if gt_points.shape[0] == 0:
        # No GT at all → all errors remain zero
        return bin_centers, mean_errors, counts

    # Build KD-tree for predicted points
    tree = cKDTree(pred_points[:, :3])

    # NN distances for GT points
    dists, _ = tree.query(gt_points[:, :3], k=1)

    xs = gt_points[:, 0]
    bin_idx = np.digitize(xs, bins) - 1

    # Keep only valid bin indices
    valid = (bin_idx >= 0) & (bin_idx < num_bins)
    dists = dists[valid]
    bin_idx = bin_idx[valid]

    # Aggregate
    np.add.at(mean_errors, bin_idx, dists)
    np.add.at(counts, bin_idx, 1)

    # Mean error per bin, bins with 0 count stay 0 automatically
    nonzero = counts > 0
    mean_errors[nonzero] = mean_errors[nonzero] / counts[nonzero]

    return bin_centers, mean_errors, counts


def modality_error_vs_range_numpy_with_zero(
    radar_points,
    lidar_points,
    pred_points,
    num_bins=50,
    x_min=None,
    x_max=None,
):
    """
    Compute per-bin NN error for radar/LiDAR separately.
    If a bin has no GT points (radar or LiDAR), its error is 0.
    """
    # Determine shared x-range
    all_x = np.concatenate([
        radar_points[:, 0] if radar_points.size > 0 else np.array([]),
        lidar_points[:, 0] if lidar_points.size > 0 else np.array([])
    ])

    if x_min is None:
        x_min = float(all_x.min()) if all_x.size > 0 else 0.0
    if x_max is None:
        x_max = float(all_x.max()) if all_x.size > 0 else 1.0

    # Radar curve
    bx_radar, radar_err, radar_cnt = nn_error_vs_x_numpy_with_zero(
        radar_points, pred_points, num_bins, x_min, x_max
    )
    # LiDAR curve
    bx_lidar, lidar_err, lidar_cnt = nn_error_vs_x_numpy_with_zero(
        lidar_points, pred_points, num_bins, x_min, x_max
    )

    return bx_radar, radar_err, lidar_err, radar_cnt, lidar_cnt

import matplotlib.pyplot as plt

def plot_mapping_error_cdf(
    radar_dists,
    lidar_dists,
    unit="m",
    title="CDF - Error",
    save_path=None,
    ax=None,
):
    """
    Plot CDF of NN mapping error for radar and LiDAR, similar to Fig. 5.

    Args:
        radar_points: (Nr, >=3) numpy array [x, y, z, ...]
        lidar_points: (Nl, >=3) numpy array [x, y, z, ...]
        pred_points:  (Np, >=3) numpy array [x, y, z, ...]
        unit:         "m" (meters) or "cm" (centimeters)
        title:        figure title
        show:         if True, calls plt.show()
        ax:           optional matplotlib axis to draw on

    Returns:
        fig, ax: matplotlib Figure and Axes objects
        (radar_x, radar_y): CDF for radar
        (lidar_x, lidar_y): CDF for LiDAR
    """

    # Convert units
    scale = 100.0 if unit == "cm" else 1.0
    radar_vals = radar_dists * scale
    lidar_vals = lidar_dists * scale

    # Compute CDFs only from points that exist
    radar_x, radar_y = make_empirical_cdf(radar_vals)
    lidar_x, lidar_y = make_empirical_cdf(lidar_vals)

    radar_x, radar_y = np.concatenate(([0.0], radar_x)), np.concatenate(([0.0], radar_y))
    lidar_x, lidar_y = np.concatenate(([0.0], lidar_x)), np.concatenate(([0.0], lidar_y))

    # Plot
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if radar_x.size > 0:
        ax.plot(radar_x, radar_y, label="Radar")
    if lidar_x.size > 0:
        ax.plot(lidar_x, lidar_y, label="LiDAR")

    ax.set_xlabel("Error ({})".format(unit))
    ax.set_ylabel("CDF")
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 8])
    ax.grid(True)
    ax.legend()
    ax.set_title(title)

    
    plt.tight_layout()
    plt.savefig(save_path)

    return fig, ax, (radar_x, radar_y), (lidar_x, lidar_y)

def make_empirical_cdf(errors):
    """
    Compute empirical CDF for a 1D array of per-point errors.
    Ensures that bins with zero counts are naturally excluded.
    """
    errors = np.asarray(errors)
    errors = errors[errors > 0]  # optional: remove exactly zero if desired

    if errors.size == 0:
        return np.array([]), np.array([])

    x = np.sort(errors)
    n = x.size
    y = np.arange(1, n + 1, dtype=float) / n
    return x, y

