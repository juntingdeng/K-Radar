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
        self.out_ch = K*F # used to be K + K*F

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
            SubMConv3d(base_ch, K*(F+1), kernel_size=1, bias=True, indice_key="head_feat")  # dx,dy,dz,i
        )
        # self.occ_head  = spconv.SparseSequential(
        #     SubMConv3d(base_ch, K, kernel_size=1, bias=True, indice_key="head_occ")   # p (logit)
        # )
        # self.ints_head  = spconv.SparseSequential(
        #     SubMConv3d(base_ch, K, kernel_size=1, bias=True, indice_key="head_occ")   # p (logit)
        # )

        self.occ_act = nn.Sigmoid()
        self.ints_act = nn.ReLU()

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
        # ints = self.ints_head(c0)
        # occ = occ.replace_feature(self.occ_act(occ.features))
        # ints = ints.replace_feature(self.ints_act(ints.features))

        N = pred.features.size(0)
        C = self.out_ch
        K, F = self.K, self.F
        feats = pred.features
        # split into slots
        feats = feats.view(feats.shape[0], self.K, -1) # [N, K, 4+1 (x, y, z, i)]
        logits = feats[:, :, 4:5] #feats[:, 0:K]                               # [N, K]
        attrs  = feats[:, :, :4] #feats[:, K: K+K*F].view(N, K, F)     # [N, K, F]
        return {"st": pred, "logits": logits, "attrs": attrs}

def _hash_zyx(b,z,y,x, Z,Y,X):
    return (((b * Z + z) * Y + y) * X + x)

def build_hash(indices, spatial_shape):
    Z, Y, X = spatial_shape
    b,z,y,x = indices[:,0], indices[:,1], indices[:,2], indices[:,3]
    return _hash_zyx(b,z,y,x, Z,Y,X)

def make_offsets(R):
    rng = torch.arange(-R, R+1, 10)
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

import torch
from spconv.pytorch import SparseConvTensor

def local_match_closest(radar: SparseConvTensor, lidar: SparseConvTensor):
    """
    For each radar voxel, find the closest LiDAR voxel (in voxel index space)
    within the same batch and use it as ground truth.

    Returns:
        matched_mask: [Nr] bool, True if a LiDAR voxel exists in that batch
        gt_offsets:   [Nr, 3] float, (dz, dy, dx) in voxel units
        gt_feat:      [Nr, C_lidar] float, LiDAR features of the matched voxel
    """
    assert tuple(radar.spatial_shape) == tuple(lidar.spatial_shape)
    device = radar.indices.device

    r_idx = radar.indices        # [Nr, 4] = [b, z, y, x]
    l_idx = lidar.indices        # [Nl, 4]
    Nr = r_idx.size(0)
    C_lidar = lidar.features.size(1)

    topk=10
    # outputs
    matched_mask = torch.zeros(Nr, dtype=torch.bool, device=device)
    gt_offsets = torch.zeros((Nr, topk, 3), dtype=radar.features.dtype, device=device)
    gt_feat = torch.zeros((Nr, topk, C_lidar), dtype=lidar.features.dtype, device=device)

    # process per batch
    batch_ids = r_idx[:, 0].unique()
    for b in batch_ids:
        r_mask = (r_idx[:, 0] == b)
        l_mask = (l_idx[:, 0] == b)

        r_inds_b = r_mask.nonzero(as_tuple=False).squeeze(1)  # global radar rows
        l_inds_b = l_mask.nonzero(as_tuple=False).squeeze(1)  # global lidar rows

        if l_inds_b.numel() == 0:
            # no lidar points in this batch → leave as unmatched
            continue

        # positions in voxel index space: [Nb_radar, 3], [Nb_lidar, 3]
        r_pos = r_idx[r_inds_b, 1:].to(torch.float32)  # (z,y,x)
        l_pos = l_idx[l_inds_b, 1:].to(torch.float32)

        # pairwise distances: [Nb_radar, Nb_lidar]
        # You can change p=1 for L1 / Manhattan distance if you prefer
        dist = torch.cdist(r_pos, l_pos, p=2)   
        # print(f'dist shape:{dist.shape}')

        # nn_dist_, nn_idx_ = dist.min(dim=1)              # [Nb_radar]
        nn_dist, nn_idx = torch.topk(dist, k=topk, dim=1, largest=False)
        # print(f'+++++++++++++++++nn_idx_: {nn_idx_}')
        # print(f'+++++++++++++++++nn_idx: {nn_idx.shape}')
        # map local LiDAR indices back to global rows
        l_rows = l_inds_b[nn_idx]                      # [Nb_radar]

        # mark matched (always True here since we have at least one lidar in batch)
        matched_mask[r_inds_b] = True

        # gt features
        gt_feat[r_inds_b] = lidar.features[l_rows]

        # offsets in voxel units: lidar - radar
        # offs = (l_idx[l_rows, 1:] - r_idx[r_inds_b, 1:]).to(radar.features.dtype)  # [Nb_radar, 3]
        offs = (l_idx[l_rows, 1:] - r_idx[r_inds_b, 1:].unsqueeze(1).repeat(1, topk, 1)).to(radar.features.dtype)  # [Nb_radar, 3]
        gt_offsets[r_inds_b] = offs

    # return matched_mask, gt_offsets, gt_feat
    return matched_mask.unsqueeze(1).repeat(1, topk).reshape(-1), gt_offsets.reshape(-1, 3), gt_feat.reshape(-1, 20)

import torch.nn.functional as F
import torch.nn as nn

class SynthLocalLoss(nn.Module):
    def __init__(self, w_occ=0.2, w_off=1.0, w_feat=1.0):
        super().__init__()
        self.w_occ, self.w_off, self.w_feat = w_occ, w_off, w_feat
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, attrs, pred_st: SparseConvTensor,
                radar_st: SparseConvTensor, lidar_st: SparseConvTensor, R=1, origin=[0,0,0], vsize_xyz=[0,0,0]):
        # pred_feat_st.features: [Nr, 4] = dx,dy,dz,i
        # pred_occ_st.features : [Nr, 1]

        # logits: [N, K, 1]
        # attrs: [N, K, 4], dx,dy,dz,i
        logits = logits.unsqueeze(1).repeat(1, 10, 1, 1).reshape(-1, logits.shape[-2], logits.shape[-1])
        attrs = attrs.unsqueeze(1).repeat(1, 10, 1, 1).reshape(-1, attrs.shape[-2], attrs.shape[-1])
        Nr = attrs.size(0)
        matched, gt_d, gt_f = local_match_closest(radar_st, lidar_st) #gt_d: zyx
        matched = matched.unsqueeze(1).repeat(1, 5)
        # print(f'matched:{matched.shape}, gt_d:{gt_d.shape}, gt_f:{gt_f.shape}, attrs:{attrs.shape}')
        gt_d = torch.flip(gt_d, dims=[1]).unsqueeze(1).repeat(1, 5, 1) #gt_d: zyx -> xyz
        gt_f = gt_f.view(gt_f.shape[0], -1, 4)
        # print(f'matched:{matched.shape}, gt_d:{gt_d.shape}, gt_f:{gt_f.shape}, attrs:{attrs.shape}')
        
        # Occupancy on all radar rows
        # print(f'here: {occ_logits.shape[0]}, {matched.shape[0]}')
        occ_loss = self.bce(logits.squeeze(), matched.float()) if Nr > 0 else logits.sum()*0.0

        # Offsets & feature on matched only
        if matched.any():
            # print("!!matched!!")
            pred_d = attrs[matched][:, :3] # xyz
            pred_i = attrs[matched][:, 3: 4]
            # print(f'pred_d: {pred_d.shape}, gt_d[matched]: {gt_d[matched].shape}')
            off_loss = F.smooth_l1_loss(pred_d, gt_d[matched])
            # feat_loss = F.l1_loss(pred_i, gt_f[matched][:, 3:4])  # if your LiDAR feat is intensity in channel 0

            offs = attrs[:, :, :3]
            voxel_center_xyz = origin + (torch.flip(radar_st.indices[:, 1:4].float(), dims=[1]) + 0.5) * vsize_xyz  # grid center
            pred_offset_m = offs * vsize_xyz.to('cuda')  # scale voxel-units → meters
            voxel_center_xyz = voxel_center_xyz.unsqueeze(1).repeat(1, 5, 1)
            voxel_center_xyz = voxel_center_xyz.unsqueeze(1).repeat(1, 10, 1, 1)
            voxel_center_xyz = voxel_center_xyz.reshape(-1, 5, voxel_center_xyz.shape[-1])
            # print(voxel_center_xyz.shape, pred_offset_m.shape)
            attrs = torch.cat([voxel_center_xyz + pred_offset_m, attrs[:, :, 3:4]], dim=-1)
            # print(f'attrs:{attrs.shape}')
            feat_loss = F.l1_loss(attrs[matched], gt_f[matched])
            
        else:
            off_loss = attrs[matched][:, :, :3].sum()*0.0
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


