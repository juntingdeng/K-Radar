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
    def __init__(self, in_ch=20, base_ch=32, K=5, F=4):
        super().__init__()
        C = base_ch
        self.K, self.F = K, F
        # outputs: [exist K]+ [attrs K*F]
        self.out_ch = K*F

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
            SubMConv3d(base_ch, self.out_ch, kernel_size=1, bias=True, indice_key="head_feat")  # dx,dy,dz,i
        )
        self.occ_head  = spconv.SparseSequential(
            SubMConv3d(base_ch, 1, kernel_size=1, bias=True, indice_key="head_occ")   # p (logit)
        )

        self.act_head = nn.Sigmoid()

    # @staticmethod
    # def _cat_if_same_coords(a: SparseConvTensor, b: SparseConvTensor):
    #     # Ensure coordinates/shape align before concatenation
    #     if not (a.indices.shape == b.indices.shape and torch.equal(a.indices, b.indices)):
    #         raise RuntimeError("Skip connection coords mismatch; check indice_key/stride path.")
    #     feats = torch.cat([a.features, b.features], dim=1)
    #     return SparseConvTensor(
    #         feats, a.indices, a.spatial_shape, a.batch_size, grid=a.grid
    #     )
    @staticmethod
    def _cat_if_same_coords(a: SparseConvTensor, b: SparseConvTensor):
        if not (a.indices.shape == b.indices.shape and torch.equal(a.indices, b.indices)):
            raise RuntimeError("Skip connection coords mismatch; check indice_key/stride path.")
        feats = torch.cat([a.features, b.features], dim=1)
        # Preserve indices, spatial_shape, *and* indice_dict
        return a.replace_feature(feats)

    def forward(self, x: SparseConvTensor):
        # Encoder
        e0 = self.enc0(x)                       # subm @ full res
        d0 = self.down0(e0)
        d0 = d0.replace_feature(self.act0(self.bn0(d0.features)))

        e1 = self.enc1(d0)
        d1 = self.down1(e1)
        d1 = d1.replace_feature(self.act1(self.bn1(d1.features)))

        e2 = self.enc2(d1)
        d2 = self.down2(e2)
        d2 = d2.replace_feature(self.act2(self.bn2(d2.features)))

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

        # occ = self.occ_head(c0)           # logits per active voxel
        pred = self.pred_head(c0)
        # pred = pred.replace_feature(self.act_head(pred.features))

        N = pred.features.size(0)
        C = self.out_ch
        K, F = self.K, self.F
        feats = pred.features
        # split into slots
        logits = feats[:, 0:K]                               # [N, K]
        # offs   = feats[:, K:K+3*K].view(N, K, 3)             # [N, K, 3] (voxel units)
        attrs  = feats[:, K: K+K*F].view(N, K, F)     # [N, K, F]
        return {"st": pred, "logits": logits, "attrs": attrs}

def _hash_zyx(b,z,y,x, Z,Y,X):
    return (((b * Z + z) * Y + y) * X + x)

def build_hash(indices, spatial_shape):
    Z, Y, X = spatial_shape
    b,z,y,x = indices[:,0], indices[:,1], indices[:,2], indices[:,3]
    return _hash_zyx(b,z,y,x, Z,Y,X)

def make_offsets(R):
    rng = torch.arange(-R, R+1, 1)
    dz, dy, dx = torch.meshgrid(rng, rng, rng, indexing="ij")
    offs = torch.stack([dz.reshape(-1), dy.reshape(-1), dx.reshape(-1)], dim=1)  # [K,3], K=(2R+1)^3
    return offs

def _safe_membership_via_searchsorted(sorted_keys: torch.Tensor,
                                      query: torch.Tensor):
    """
    sorted_keys: 1D int64, ascending, device=CUDA
    query      : 1D int64, device matches sorted_keys
    Returns: (is_member [N], pos [N], gathered_equal [N])
      - is_member: bool mask indicating sorted_keys contains query[i]
      - pos: insertion positions from searchsorted (int64)
      - gathered_equal: values gathered from sorted_keys at pos (valid only where is_member)
    """
    assert sorted_keys.dim() == 1 and query.dim() == 1
    pos = torch.searchsorted(sorted_keys, query)  # [N], int64
    # valid where 0 <= pos < len(sorted_keys)
    valid = (pos >= 0) & (pos < sorted_keys.numel())

    # gather only at valid positions
    gathered = torch.empty_like(query)
    gathered[valid] = sorted_keys[pos[valid]]

    is_member = torch.zeros_like(valid, dtype=torch.bool)
    is_member[valid] = (gathered[valid] == query[valid])
    return is_member, pos, gathered

@torch.no_grad()
def local_match(radar: SparseConvTensor, lidar: SparseConvTensor, R=1):
    assert tuple(radar.spatial_shape) == tuple(lidar.spatial_shape)
    Z, Y, X = radar.spatial_shape
    device = radar.indices.device
    # ---- build LiDAR hash and sort
    hL = build_hash(lidar.indices.to(torch.int64), (Z, Y, X)).to(device)
    hL_sorted, ordL = torch.sort(hL)  # IMPORTANT: use sorted keys for searchsorted

    # ---- enumerate radar neighbor candidates
    offs = make_offsets(R).to(device)                    # [K,3], int64
    b,z,y,x = radar.indices[:,0], radar.indices[:,1], radar.indices[:,2], radar.indices[:,3]
    K = offs.size(0)
    bz = b[:,None]
    zz = (z[:,None] + offs[None,:,0]).clamp_(0, Z-1)
    yy = (y[:,None] + offs[None,:,1]).clamp_(0, Y-1)
    xx = (x[:,None] + offs[None,:,2]).clamp_(0, X-1)

    cand_hash = _hash_zyx(bz, zz, yy, xx, Z, Y, X).reshape(-1).to(torch.int64)  # [Nr*K]

    # ---- safe membership check (NO out-of-bounds indexing)
    is_member, pos, gathered = _safe_membership_via_searchsorted(hL_sorted, cand_hash)
    hit = is_member.view(-1, K)                                           # [Nr,K]

    # choose the closest neighbor in L1 grid distance among hits
    dL1 = offs.abs().sum(dim=1)[None,:].expand(hit.size(0), K)            # [Nr,K]
    dL1_masked = torch.where(hit, dL1, torch.full_like(dL1, 1e9))
    best_k = dL1_masked.argmin(dim=1)                                     # [Nr]
    matched_mask = dL1_masked[torch.arange(hit.size(0), device=device), best_k] < 1e9

    # map (q,k) -> lidar row indices using sorted positions → original order
    flat_idx = torch.arange(hit.size(0), device=device) * K + best_k      # [Nr]
    pos_flat = pos.view(-1)[flat_idx]                                     # [Nr]
    # valid only where matched_mask
    lidar_row_sorted = pos_flat[matched_mask]
    lidar_row = ordL[lidar_row_sorted]                                    # [N_matched]

    # build outputs
    Nr = radar.indices.size(0)
    C_lidar = lidar.features.size(1)
    gt_feat = torch.zeros((Nr, C_lidar), device=device, dtype=lidar.features.dtype)
    gt_feat[matched_mask] = lidar.features[lidar_row]

    # offsets (voxel units) from radar voxel center to chosen neighbor voxel center
    sel_z = zz.view(Nr, K)[torch.arange(Nr, device=device), best_k]
    sel_y = yy.view(Nr, K)[torch.arange(Nr, device=device), best_k]
    sel_x = xx.view(Nr, K)[torch.arange(Nr, device=device), best_k]
    gt_offsets = torch.stack([
        (sel_z - z).to(radar.features.dtype),
        (sel_y - y).to(radar.features.dtype),
        (sel_x - x).to(radar.features.dtype),
    ], dim=1)  # [Nr,3]

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
        matched, gt_d, gt_f = local_match(radar_st, lidar_st, R=R) #gt_d: zyx
        gt_d = torch.flip(gt_d, dims=[1]) #gt_d: zyx -> xyz

        # Occupancy on all radar rows
        occ_logits = pred_occ_st.squeeze(1)
        # print(f'here: {occ_logits.shape[0]}, {matched.shape[0]}')
        occ_loss = self.bce(occ_logits.mean(-1), matched.float()) if Nr > 0 else occ_logits.sum()*0.0

        # Offsets & feature on matched only
        if matched.any():
            # print("!!matched!!")
            pred_d = pred_feat_st.features[:, :3][matched] # xyz
            pred_i = pred_feat_st.features[:, 3:4][matched]
            off_loss = F.smooth_l1_loss(pred_d, gt_d[matched])
            feat_loss = F.l1_loss(pred_i, gt_f[matched, 3:4])  # if your LiDAR feat is intensity in channel 0
        else:
            off_loss = pred_feat_st.features.sum()*0.0
            feat_loss = off_loss
        # print(f'occ_loss:{occ_loss}, off_loss:{off_loss}, feat_loss:{feat_loss}')
        return self.w_occ*occ_loss + self.w_off*off_loss + self.w_feat*feat_loss

def voxel_center(cfg):
    # z_id, y_id, x_id, z_c, y_c, x_c
    x_min, y_min, z_min, x_max, y_max, z_max = cfg.DATASET.roi.xyz
    vsize_xyz = cfg.DATASET.roi.voxel_size
    x_size = int(round((x_max-x_min)/vsize_xyz[0]))
    y_size = int(round((y_max-y_min)/vsize_xyz[1]))
    z_size = int(round((z_max-z_min)/vsize_xyz[2]))

    centers = torch.zeros((z_size, y_size, x_size, 3))
    for zi in range(z_size):
        z_c = z_min + vsize_xyz[2]*zi + vsize_xyz[2]/2
        for yi in range(y_size):
            y_c = y_min + vsize_xyz[1]*yi + vsize_xyz[1]/2
            for xi in range(z_size):
                x_c = x_min + vsize_xyz[0]*xi + vsize_xyz[0]/2

                centers[zi][yi][xi] = torch.tensor([z_c, y_c, x_c])
    
    return centers

import torch
from spconv.pytorch import SparseConvTensor

def scatter_radar_to_union(radar_st, union_idx, spatial_shape, batch_size, fill_value=0.0):
    """
    radar_st: SparseConvTensor with features at radar indices
    union_idx: [M,4] LongTensor = union of radar & lidar voxel coords (b,z,y,x)
    spatial_shape: same as radar_st.spatial_shape  (Z,Y,X)
    batch_size: same as radar_st.batch_size
    fill_value: what to put in voxels that exist in union but not in radar
    """
    device = radar_st.indices.device
    dtype_idx = torch.int64

    radar_idx = radar_st.indices.to(dtype_idx).to(device)   # [Nr,4]
    union_idx = union_idx.to(dtype_idx).to(device)          # [M,4]
    Nr = radar_idx.shape[0]
    M  = union_idx.shape[0]

    ZMAX, YMAX, XMAX = spatial_shape
    BMAX = batch_size

    def hash_idx(idx):
        b, z, y, x = idx[:,0], idx[:,1], idx[:,2], idx[:,3]
        return (((b * ZMAX + z) * YMAX + y) * XMAX + x)

    # 1) hashes
    radar_hash = hash_idx(radar_idx)      # [Nr]
    union_hash = hash_idx(union_idx)      # [M]

    # 2) sort radar hashes for searchsorted
    sorted_h, sorted_ord = torch.sort(radar_hash)           # [Nr], [Nr]
    sorted_feat = radar_st.features[sorted_ord]             # [Nr,C]
    C = sorted_feat.shape[1]

    # 3) searchsorted to find potential positions of union_hash in sorted_h
    pos = torch.searchsorted(sorted_h, union_hash)          # [M], int64

    # 4) SAFE membership test (no OOB indexing)
    #    Only gather where 0 <= pos < Nr
    valid_pos = (pos >= 0) & (pos < Nr)                     # [M] bool

    gathered = torch.empty_like(union_hash)                 # [M]
    # fill with sentinel that cannot equal any real hash
    gathered[:] = -1
    gathered[valid_pos] = sorted_h[pos[valid_pos]]

    is_member = valid_pos & (gathered == union_hash)        # [M] bool

    # 5) Build union feature matrix
    feat_dim = C
    fill = torch.full((1, feat_dim),
                      float(fill_value),
                      device=radar_st.features.device,
                      dtype=radar_st.features.dtype)

    union_feat = torch.empty((M, feat_dim),
                             device=radar_st.features.device,
                             dtype=radar_st.features.dtype)

    # copy radar features where there is a match
    idx_radar_rows = pos[is_member]                         # indices into sorted_feat
    union_feat[is_member] = sorted_feat[idx_radar_rows]

    # fill zeros (or fill_value) for LiDAR-only voxels
    union_feat[~is_member] = fill

    # 6) Construct new SparseConvTensor on the union support
    union_st = SparseConvTensor(
        features=union_feat,
        indices=union_idx.int(),
        spatial_shape=spatial_shape,
        batch_size=batch_size,
    )
    return union_st


