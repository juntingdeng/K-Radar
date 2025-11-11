import os
import numpy as np
import matplotlib.pyplot as plt

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion

# Gaussian upsampling params
FC_HZ = 77e9
C = 299792458.0
LAMBDA = C / FC_HZ

# Effective apertures (tune to your radar rig; these are reasonable starters)
D_AZ = 0.12   # meters (azimuth aperture)
D_EL = 0.02   # meters (elevation aperture)

SIGMA_R = C / (2 * 3e9)   # range sigma from bandwidth ~3 GHz -> ~0.05 m
NS_PER_RADAR_POINT = 20   # how many samples per radar detection

def gaussian_sigma_from_aperture(lmbda, D):
    """
    Convert array aperture D to Gaussian power sigma via HPBW≈0.886*λ/D and FWHM=2.355σ.
    Returns sigma (radians) for the **power** Gaussian.
    """
    hpbw = 0.886 * lmbda / D
    sigma = hpbw / 2.355
    return sigma

def gaussian_upsample_radar(radar_xyz, Ns=20, sigma_r=0.05, lmbda=LAMBDA, D_az=D_AZ, D_el=D_EL, rng=None):
    """
    Radar -> LiDAR-like: Gaussian upsampling.
    Treat each radar point as the center of a Gaussian in 3D whose angular spreads come from λ/D.
    - radar_xyz: (3xN) in ego frame
    - Ns: samples per radar point
    - sigma_r: range std (m)
    - D_az, D_el: effective apertures (m)
    """
    if radar_xyz.size == 0:
        return np.zeros((3,0))

    if rng is None:
        rng = np.random.default_rng(0)

    sigma_az = gaussian_sigma_from_aperture(lmbda, D_az)
    sigma_el = gaussian_sigma_from_aperture(lmbda, D_el)

    x, y, z = radar_xyz
    r = np.sqrt(x**2 + y**2 + z**2) + 1e-9
    az = np.arctan2(y, x)
    el = np.arcsin(z / r)

    # Build covariance per detection in local (range, az, el) terms, then sample & map to Cartesian
    out = []
    for ri, azi, eli in zip(r, az, el):
        # Draw (Δr, Δaz, Δel) ~ N(0, diag([σ_r^2, σ_az^2, σ_el^2]))
        dr  = rng.normal(0.0, sigma_r, size=Ns)
        daz = rng.normal(0.0, sigma_az, size=Ns)
        delv= rng.normal(0.0, sigma_el, size=Ns)

        r_s  = np.maximum(ri + dr, 0.0)
        az_s = azi + daz
        el_s = eli + delv

        xs = r_s * np.cos(el_s) * np.cos(az_s)
        ys = r_s * np.cos(el_s) * np.sin(az_s)
        zs = r_s * np.sin(el_s)
        out.append(np.stack([xs, ys, zs], axis=0))

    return np.hstack(out)

# ---- insert this helper somewhere near the top of rdr_sparse_preprocessor.py ----
import numpy as np
import torch

# ---- insert this helper somewhere near the top of rdr_sparse_preprocessor.py ----
import numpy as np

def upsample_rdr_sparse_xyzpw(
    pts_xyzw,                      # (N,4) float32: [x, y, z, pw]
    base_samples_per_point=8,      # baseline samples per original point
    sigma_xyz=(0.30, 0.30, 0.50),  # Gaussian std (m) along x/y/z
    bev_cell=(0.5, 0.5),           # (dy, dx) for BEV density modulation in meters
    bev_extent=((-60., 60.), (0., 120.)),  # ((y_min,y_max),(x_min,x_max)) meters
    density_exponent=1.0,          # 0→ignore BEV; >1 favors dense cells; <1 flattens
    include_original=True,         # keep the original points
    power_mode="multiply",         # "multiply" (pw * kernel) or "copy" (unchanged pw)
    kernel="gaussian",             # "gaussian" or "sinc2"
    sinc_w=1.0,                    # main-lobe width for sinc2 kernel, in meters (if used)
    power_noise_std=0.0,           # optional additive noise on power
    min_power=None, max_power=None,# optional power clipping
    rng=None
):
    """
    Returns:
        (M,4) float32 array of [x, y, z, pw] after upsampling.

    Notes:
    - Gaussian kernel: k = exp(-0.5 * (dx^2/sx^2 + dy^2/sy^2 + dz^2/sz^2))
    - Sinc^2 kernel  : k = sinc((dx)/w)^2 * sinc((dy)/w)^2   [z left Gaussian]
      Use this when you want a physically-inspired lateral spread; tune 'sinc_w'.
    """
    if rng is None:
        rng = np.random.default_rng()

    pts = np.asarray(pts_xyzw, dtype=np.float32)
    assert pts.ndim == 2 and pts.shape[1] == 4, "pts must be (N,4) [x,y,z,pw]"
    N = pts.shape[0]
    if N == 0:
        return pts

    sx, sy, sz = map(float, sigma_xyz)
    y_min, y_max = bev_extent[0]
    x_min, x_max = bev_extent[1]
    cy, cx = bev_cell
    nx = int(np.ceil((x_max - x_min) / cx))
    ny = int(np.ceil((y_max - y_min) / cy))

    x = pts[:, 0]; y = pts[:, 1]
    ix = ((x - x_min) / cx).astype(int)
    iy = ((y - y_min) / cy).astype(int)
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)

    # --- BEV density map (for sample count modulation) ---
    if np.any(valid):
        counts = np.zeros((ny, nx), dtype=np.int32)
        for j in np.where(valid)[0]:
            counts[iy[j], ix[j]] += 1
        c = np.zeros(N, dtype=np.float32)
        v = np.where(valid)[0]
        c[v] = counts[iy[v], ix[v]].astype(np.float32)
        if c.max() > 0:
            c = c / c.max()
        weights = np.power(np.clip(c, 1e-6, 1.0), density_exponent).astype(np.float32)
    else:
        weights = np.ones(N, dtype=np.float32)

    samples_per_point = np.maximum(0, np.round(base_samples_per_point * weights)).astype(int)
    if samples_per_point.sum() == 0:
        samples_per_point = np.ones(N, dtype=int)

    total_new = int(samples_per_point.sum())
    out_sz = total_new + (N if include_original else 0)
    out = np.empty((out_sz, 4), dtype=np.float32)

    # precalc inverses for Gaussian
    inv2_sx2 = 1.0 / (2.0 * sx * sx) if sx > 0 else np.inf
    inv2_sy2 = 1.0 / (2.0 * sy * sy) if sy > 0 else np.inf
    inv2_sz2 = 1.0 / (2.0 * sz * sz) if sz > 0 else np.inf

    write = 0
    use_gauss = (kernel.lower() == "gaussian")
    use_sinc2 = (kernel.lower() == "sinc2")
    sinc_w = float(sinc_w) if sinc_w is not None else 1.0

    for i in range(N):
        k = samples_per_point[i]
        if k <= 0:
            continue
        xi, yi, zi, pwi = pts[i]

        dx = rng.normal(0.0, sx, size=k).astype(np.float32)
        dy = rng.normal(0.0, sy, size=k).astype(np.float32)
        dz = rng.normal(0.0, sz, size=k).astype(np.float32)

        xs = xi + dx
        ys = yi + dy
        zs = zi + dz

        if use_gauss:
            kval = np.exp(-(dx*dx)*inv2_sx2 - (dy*dy)*inv2_sy2 - (dz*dz)*inv2_sz2).astype(np.float32)
        elif use_sinc2:
            # numpy.sinc(u) = sin(pi*u)/(pi*u)
            # Apply sinc^2 laterally; keep Gaussian along z to avoid infinite support
            kx = np.sinc(dx / max(sinc_w, 1e-6))
            ky = np.sinc(dy / max(sinc_w, 1e-6))
            kz = np.exp(-(dz*dz)*inv2_sz2)  # gentle confinement in z
            kval = (kx*kx * ky*ky * kz).astype(np.float32)
        else:
            raise ValueError("kernel must be 'gaussian' or 'sinc2'")

        if power_mode == "multiply":
            ps = (pwi * kval).astype(np.float32)
        elif power_mode == "copy":
            ps = np.full(k, pwi, dtype=np.float32)
        else:
            raise ValueError("power_mode must be 'multiply' or 'copy'")

        if power_noise_std and power_noise_std > 0:
            ps += rng.normal(0.0, power_noise_std, size=k).astype(np.float32)

        if (min_power is not None) or (max_power is not None):
            lo = -np.inf if min_power is None else min_power
            hi =  np.inf if max_power is None else max_power
            ps = np.clip(ps, lo, hi, out=ps)

        out[write:write+k, 0] = xs
        out[write:write+k, 1] = ys
        out[write:write+k, 2] = zs
        out[write:write+k, 3] = ps
        write += k

    if include_original:
        out[write:write+N] = pts
        write += N

    return out[:write]

@torch.no_grad()
# def upsample_rdr_sparse_xyzpw_torch(
#     pts_xyzw: torch.Tensor,             # (N,4) tensor: [x, y, z, pw]
#     base_samples_per_point: int = 8,    # baseline samples per original point
#     sigma_xyz=(0.30, 0.30, 0.50),       # Gaussian std (m) along x/y/z
#     bev_cell=(0.5, 0.5),                # (dy, dx) in meters for BEV density modulation
#     bev_extent=((-60., 60.), (0., 120.)),  # ((y_min,y_max),(x_min,x_max)) in meters
#     density_exponent: float = 1.0,      # 0→ignore BEV; >1 favors dense cells; <1 flattens
#     include_original: bool = True,      # keep the original points
#     power_mode: str = "multiply",       # "multiply" (pw * kernel) or "copy"
#     kernel: str = "gaussian",           # "gaussian" or "sinc2"
#     sinc_w: float = 1.0,                # main-lobe width (m) for sinc² (lateral); z stays Gaussian
#     power_noise_std: float = 0.0,       # optional additive power noise
#     min_power: float = None,     # optional power clipping
#     max_power: float = None,     # optional power clipping
#     generator: torch.Generator = None,  # for determinism if desired
# ) -> torch.Tensor:
def upsample_rdr_sparse_xyzpw_torch(
    pts_xyzw: torch.Tensor,
    base_samples_per_point: int = 20,          # moderate, avoids overfilling
    sigma_xyz=(0.60, 0.90, 0.60),             # [m] ≈ (range, cross-range, vertical)
    bev_cell=(0.4, 0.4),                      # [m] BEV grid (dy, dx)
    bev_extent=((-40., 40.), (0., 120.)),     # [m] ((ymin,ymax), (xmin,xmax))
    density_exponent: float = 1.0,            # respect original BEV density
    include_original: bool = True,
    power_mode: str = "multiply",             # degrade pw with kernel
    kernel: str = "sinc2",                    # RETINA has ~1° angular bins → use sinc²
    sinc_w: float = 1.0,                      # [m] ~ cross-range main-lobe near 50–60 m
    power_noise_std: float = 0.0,
    min_power: float = None,            # keep power non-negative
    max_power: float = None,
    generator: torch.Generator = None,
) -> torch.Tensor:
    """
    Returns:
        (M,4) tensor [x, y, z, pw] after upsampling, same dtype/device as input.

    Notes:
    - Gaussian kernel: k = exp(-0.5 * (dx^2/sx^2 + dy^2/sy^2 + dz^2/sz^2))
    - Sinc^2 kernel  : k = sinc(dx/w)^2 * sinc(dy/w)^2 * exp(-0.5 * dz^2/sz^2)
      (torch.sinc uses normalized sinc: sin(pi x)/(pi x))
    """
    assert pts_xyzw.ndim == 2 and pts_xyzw.shape[-1] == 4, "pts_xyzw must be (N,4)"
    N = pts_xyzw.shape[0]
    if N == 0:
        return pts_xyzw

    device = pts_xyzw.device
    dtype  = pts_xyzw.dtype

    sx, sy, sz = map(float, sigma_xyz)
    cy, cx = bev_cell
    (y_min, y_max), (x_min, x_max) = bev_extent

    x = pts_xyzw[:, 0]
    y = pts_xyzw[:, 1]
    z = pts_xyzw[:, 2]
    pw = pts_xyzw[:, 3]

    # --------------------------
    # 1) BEV density (to modulate sample counts)
    # --------------------------
    nx = int(torch.ceil(torch.tensor((x_max - x_min) / cx)).item())
    ny = int(torch.ceil(torch.tensor((y_max - y_min) / cy)).item())

    ix = torch.floor((x - x_min) / cx).to(torch.int64)
    iy = torch.floor((y - y_min) / cy).to(torch.int64)

    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    weights = torch.ones(N, device=device, dtype=torch.float32)

    if torch.any(valid):
        # 2D histogram via linear indices + bincount
        lin = (iy[valid] * nx + ix[valid]).to(torch.int64)
        counts_lin = torch.bincount(lin, minlength=ny * nx).to(torch.float32)
        # pull each point's BEV count
        c = torch.zeros(N, device=device, dtype=torch.float32)
        c_valid = counts_lin[lin]  # count per valid point’s cell
        c[valid] = c_valid
        # normalize to [0,1]
        cmax = torch.max(c)
        if cmax > 0:
            c = c / cmax
        # weights = c^(density_exponent) with floor to avoid zeros
        weights = torch.clamp(c, min=1e-6).pow(float(density_exponent))

    # --------------------------
    # 2) Per-point sample counts
    # --------------------------
    base = torch.tensor(float(base_samples_per_point), device=device)
    spp = torch.round(base * weights).to(torch.int64)
    if torch.sum(spp) == 0:
        spp = torch.ones_like(spp)

    total_new = int(torch.sum(spp).item())

    # --------------------------
    # 3) Build repeated seeds and sample jitters (vectorized)
    # --------------------------
    # Repeat seed points by spp
    repeat_idx = torch.repeat_interleave(torch.arange(N, device=device), repeats=spp, dim=0)
    xi = x[repeat_idx]
    yi = y[repeat_idx]
    zi = z[repeat_idx]
    pwi = pw[repeat_idx]

    # Random jitters
    # Use dtype float32 for noise, then cast to input dtype after
    g = generator
    dx = torch.randn(total_new, device=device, generator=g) * sx
    dy = torch.randn(total_new, device=device, generator=g) * sy
    dz = torch.randn(total_new, device=device, generator=g) * sz

    xs = (xi + dx).to(dtype)
    ys = (yi + dy).to(dtype)
    zs = (zi + dz).to(dtype)

    # --------------------------
    # 4) Kernel values (Gaussian or sinc^2)
    # --------------------------
    kernel_l = kernel.lower()
    if kernel_l == "gaussian":
        inv2_sx2 = float('inf') if sx <= 0 else 1.0 / (2.0 * sx * sx)
        inv2_sy2 = float('inf') if sy <= 0 else 1.0 / (2.0 * sy * sy)
        inv2_sz2 = float('inf') if sz <= 0 else 1.0 / (2.0 * sz * sz)
        kval = torch.exp(-(dx * dx) * inv2_sx2 - (dy * dy) * inv2_sy2 - (dz * dz) * inv2_sz2).to(dtype)
    elif kernel_l == "sinc2":
        w = max(float(sinc_w), 1e-6)
        kx = torch.sinc(dx / w)
        ky = torch.sinc(dy / w)
        inv2_sz2 = float('inf') if sz <= 0 else 1.0 / (2.0 * sz * sz)
        kz = torch.exp(-(dz * dz) * inv2_sz2)
        kval = (kx * kx * ky * ky * kz).to(dtype)
    else:
        raise ValueError("kernel must be 'gaussian' or 'sinc2'")

    # --------------------------
    # 5) Power handling
    # --------------------------
    if power_mode == "multiply":
        ps = (pwi.to(dtype) * kval)
    elif power_mode == "copy":
        ps = pwi.to(dtype)
    else:
        raise ValueError("power_mode must be 'multiply' or 'copy'")

    if power_noise_std and power_noise_std > 0:
        ps = ps + torch.randn_like(ps, generator=g) * float(power_noise_std)

    if (min_power is not None) or (max_power is not None):
        lo = torch.tensor(-torch.inf, device=device, dtype=dtype) if min_power is None else torch.tensor(min_power, device=device, dtype=dtype)
        hi = torch.tensor(torch.inf, device=device, dtype=dtype)  if max_power is None else torch.tensor(max_power, device=device, dtype=dtype)
        ps = torch.clamp(ps, min=lo.item(), max=hi.item())

    new_pts = torch.stack([xs, ys, zs, ps], dim=-1)

    # --------------------------
    # 6) Append originals (optional) and return
    # --------------------------
    if include_original:
        out = torch.cat([new_pts, pts_xyzw.to(device=device, dtype=dtype)], dim=0)
    else:
        out = new_pts

    return out
