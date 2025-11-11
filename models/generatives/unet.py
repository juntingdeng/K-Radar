import torch
import torch.nn as nn
import spconv.pytorch as spconv
from spconv.pytorch import SparseConvTensor, SubMConv3d, SparseConv3d, SparseInverseConv3d

def subm_block(in_ch, out_ch, kernel_size=3, indice_key=None):
    return spconv.SparseSequential(
        SubMConv3d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size//2,
                   bias=False, indice_key=indice_key),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(inplace=True),
        SubMConv3d(out_ch, out_ch, kernel_size=kernel_size, padding=kernel_size//2,
                   bias=False, indice_key=indice_key),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(inplace=True),
    )

class SparseUNet3D(nn.Module):
    """
    3D U-Net with spconv:
      - Encoder: subm -> downsample
      - Decoder: upsample (inverse conv) -> skip concat -> subm
    Use distinct indice_key per level (e.g., "subm0", "down0", "up0").
    """
    def __init__(self, in_ch=4, base_ch=32, num_classes=20):
        super().__init__()
        C = base_ch

        # ---------------- Encoder ----------------
        self.enc0 = subm_block(in_ch, C, indice_key="subm0")
        self.down0 = SparseConv3d(C, C*2, kernel_size=2, stride=2, bias=False, indice_key="down0")
        self.bn0   = nn.BatchNorm1d(C*2); self.act0 = nn.ReLU(inplace=True)

        self.enc1 = subm_block(C*2, C*2, indice_key="subm1")
        self.down1 = SparseConv3d(C*2, C*4, kernel_size=2, stride=2, bias=False, indice_key="down1")
        self.bn1   = nn.BatchNorm1d(C*4); self.act1 = nn.ReLU(inplace=True)

        self.enc2 = subm_block(C*4, C*4, indice_key="subm2")
        self.down2 = SparseConv3d(C*4, C*8, kernel_size=2, stride=2, bias=False, indice_key="down2")
        self.bn2   = nn.BatchNorm1d(C*8); self.act2 = nn.ReLU(inplace=True)

        # bottleneck
        self.bottleneck = subm_block(C*8, C*8, indice_key="subm3")

        # ---------------- Decoder ----------------
        # Inverse conv must share the *down* indice_key to upsample onto the same grid
        self.up2 = SparseInverseConv3d(C*8, C*4, kernel_size=2, bias=False, indice_key="down2")
        self.dec2 = subm_block(C*8, C*4, indice_key="subm2")  # after cat: C*4 (up) + C*4 (skip)

        self.up1 = SparseInverseConv3d(C*4, C*2, kernel_size=2, bias=False, indice_key="down1")
        self.dec1 = subm_block(C*4, C*2, indice_key="subm1")

        self.up0 = SparseInverseConv3d(C*2, C, kernel_size=2, bias=False, indice_key="down0")
        self.dec0 = subm_block(C*2, C, indice_key="subm0")

        # Head: point-wise (submanifold) 1x1 "conv"
        self.pred_head = spconv.SparseSequential(
            SubMConv3d(base_ch, 4, kernel_size=1, bias=True, indice_key="head_feat")  # dx,dy,dz,i
        )
        self.occ_head  = spconv.SparseSequential(
            SubMConv3d(base_ch, 1, kernel_size=1, bias=True, indice_key="head_occ")   # p (logit)
        )


    @staticmethod
    def _cat_if_same_coords(a: SparseConvTensor, b: SparseConvTensor):
        # Ensure coordinates/shape align before concatenation
        if not (a.indices.shape == b.indices.shape and torch.equal(a.indices, b.indices)):
            raise RuntimeError("Skip connection coords mismatch; check indice_key/stride path.")
        feats = torch.cat([a.features, b.features], dim=1)
        return SparseConvTensor(
            feats, a.indices, a.spatial_shape, a.batch_size, grid=a.grid
        )

    def forward(self, x: SparseConvTensor):
        # Encoder
        e0 = self.enc0(x)                       # subm @ full res
        d0 = self.down0(e0); d0F = self.act0(self.bn0(d0.features))
        d0 = SparseConvTensor(d0F, d0.indices, d0.spatial_shape, d0.batch_size, grid=d0.grid)

        e1 = self.enc1(d0)
        d1 = self.down1(e1); d1F = self.act1(self.bn1(d1.features))
        d1 = SparseConvTensor(d1F, d1.indices, d1.spatial_shape, d1.batch_size, grid=d1.grid)

        e2 = self.enc2(d1)
        d2 = self.down2(e2); d2F = self.act2(self.bn2(d2.features))
        d2 = SparseConvTensor(d2F, d2.indices, d2.spatial_shape, d2.batch_size, grid=d2.grid)

        b  = self.bottleneck(d2)

        # Decoder
        u2 = self.up2(b)              # up to e2 grid (shares "down2" rulebook)
        c2 = self._cat_if_same_coords(u2, e2)
        c2 = self.dec2(c2)

        u1 = self.up1(c2)
        c1 = self._cat_if_same_coords(u1, e1)
        c1 = self.dec1(c1)

        u0 = self.up0(c1)
        c0 = self._cat_if_same_coords(u0, e0)
        c0 = self.dec0(c0)

        out = self.head(c0)           # logits per active voxel
        return out

def _hash_zyx(b,z,y,x, Z,Y,X):
    return (((b * Z + z) * Y + y) * X + x)

def build_hash(indices, spatial_shape):
    Z, Y, X = spatial_shape
    b,z,y,x = indices[:,0], indices[:,1], indices[:,2], indices[:,3]
    return _hash_zyx(b,z,y,x, Z,Y,X)

def make_offsets(R):
    rng = torch.arange(-R, R+1, dtype=torch.int64)
    dz, dy, dx = torch.meshgrid(rng, rng, rng, indexing="ij")
    offs = torch.stack([dz.reshape(-1), dy.reshape(-1), dx.reshape(-1)], dim=1)  # [K,3], K=(2R+1)^3
    return offs

@torch.no_grad()
def local_match(radar: SparseConvTensor, lidar: SparseConvTensor, R=1):
    """
    For each radar index row, pick nearest LiDAR voxel within Chebyshev radius R.
    Returns:
      matched_mask [Nr] (bool): True if a LiDAR neighbor exists
      gt_offsets   [Nr,3]      : (dz,dy,dx) from voxel center to LiDAR-center voxel (grid units)
      gt_feat      [Nr,C]      : LiDAR features at the matched voxel
    """
    assert tuple(radar.spatial_shape) == tuple(lidar.spatial_shape)
    Z,Y,X = radar.spatial_shape
    device = radar.indices.device

    # Hash LiDAR coords for O(1) membership
    h_lidar = build_hash(lidar.indices, (Z,Y,X))
    # Sort for binary search
    hL, ordL = torch.sort(h_lidar)

    offs = make_offsets(R).to(device)
    Nr = radar.indices.size(0)
    b,z,y,x = (radar.indices[:,0], radar.indices[:,1], radar.indices[:,2], radar.indices[:,3])

    # Enumerate all neighbor candidates for each radar row
    # shape: [Nr, K, 4]
    K = offs.size(0)
    bz = b[:,None]
    zz = (z[:,None] + offs[None,:,0]).clamp_(0, Z-1)
    yy = (y[:,None] + offs[None,:,1]).clamp_(0, Y-1)
    xx = (x[:,None] + offs[None,:,2]).clamp_(0, X-1)

    cand_hash = _hash_zyx(bz, zz, yy, xx, Z,Y,X).reshape(Nr*K)

    # searchsorted to test membership
    pos = torch.searchsorted(hL, cand_hash)
    hit = (pos < hL.numel()) & (hL[pos] == cand_hash)
    hit = hit.view(Nr, K)

    # If multiple hits, pick the **closest in L1 grid distance** (or first)
    dL1 = offs.abs().sum(dim=1)[None,:].expand(Nr, K)  # [Nr,K]
    dL1_masked = torch.where(hit, dL1, torch.full_like(dL1, 1e9))
    best_k = dL1_masked.argmin(dim=1)                  # [Nr]
    matched_mask = dL1_masked[torch.arange(Nr, device=device), best_k] < 1e9

    # Map best_k -> LiDAR row index
    flat_idx = (torch.arange(Nr, device=device) * K + best_k)
    pos_flat = pos.view(-1)[flat_idx]
    lidar_row = ordL[pos_flat]     # [Nr] (undefined where not matched, but masked)

    # Targets
    gt_feat = torch.zeros((Nr, lidar.features.size(1)), device=device, dtype=lidar.features.dtype)
    gt_feat[matched_mask] = lidar.features[lidar_row[matched_mask]]

    # Offsets in **voxel units** from radar voxel center to LiDAR voxel center
    dzz = (zz.view(Nr, K)[torch.arange(Nr, device=device), best_k] - z).to(radar.features.dtype)
    dyy = (yy.view(Nr, K)[torch.arange(Nr, device=device), best_k] - y).to(radar.features.dtype)
    dxx = (xx.view(Nr, K)[torch.arange(Nr, device=device), best_k] - x).to(radar.features.dtype)
    gt_offsets = torch.stack([dzz, dyy, dxx], dim=1)  # grid steps

    return matched_mask, gt_offsets, gt_feat


import torch.nn.functional as F
import torch.nn as nn

class SynthLocalLoss(nn.Module):
    def __init__(self, w_occ=0.2, w_off=1.0, w_feat=1.0):
        super().__init__()
        self.w_occ, self.w_off, self.w_feat = w_occ, w_off, w_feat
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred_feat_st: SparseConvTensor, pred_occ_st: SparseConvTensor,
                radar_st: SparseConvTensor, lidar_st: SparseConvTensor, R=1):
        # pred_feat_st.features: [Nr, 4] = dx,dy,dz,i
        # pred_occ_st.features : [Nr, 1]
        Nr = pred_feat_st.features.size(0)
        matched, gt_d, gt_f = local_match(radar_st, lidar_st, R=R)

        # Occupancy on all radar rows
        occ_logits = pred_occ_st.features.squeeze(1)
        occ_loss = self.bce(occ_logits, matched.float()) if Nr > 0 else occ_logits.sum()*0.0

        # Offsets & feature on matched only
        if matched.any():
            pred_d = pred_feat_st.features[:, :3][matched]
            pred_i = pred_feat_st.features[:, 3:4][matched]
            off_loss = F.smooth_l1_loss(pred_d, gt_d[matched])
            feat_loss = F.l1_loss(pred_i, gt_f[matched, :1])  # if your LiDAR feat is intensity in channel 0
        else:
            off_loss = pred_feat_st.features.sum()*0.0
            feat_loss = off_loss

        return self.w_occ*occ_loss + self.w_off*off_loss + self.w_feat*feat_loss