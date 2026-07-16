import torch
from spconv.pytorch import SparseConvTensor
import spconv.pytorch as spconv
from spconv.pytorch import SparseConvTensor, SubMConv3d, SparseConv3d, SparseInverseConv3d
from models.generatives.unet import subm_block

def local_match_closest_mdn(radar: SparseConvTensor, lidar: SparseConvTensor, gt_topk: int, R=None):
    """
    For each radar voxel, find top-k closest LiDAR voxels (in voxel index space),
    per batch, and return offsets/features in clean shapes.

    R: optional search radius (Euclidean, voxel-index units). Candidates farther than R
       are excluded (Sec. IV-B: "local neighborhood of candidate LiDAR correspondences
       ... within radius R"). If None, no radius filtering is applied.

    Returns:
        matched_mask:    (Nr,) bool, True iff at least one candidate is within R
        gt_offsets:      (Nr, topk, 3) float, (dz,dy,dx) in voxel units
        gt_feat:         (Nr, topk, C_lidar) float, LiDAR features for each candidate
        gt_coords:       (Nr*topk, 4) matched LiDAR voxel indices
        candidate_valid: (Nr, topk) bool, True where that candidate slot is a real,
                          within-R match (False for padding or beyond-R candidates)
    """
    assert tuple(radar.spatial_shape) == tuple(lidar.spatial_shape)
    device = radar.indices.device

    r_idx = radar.indices        # (Nr, 4) [b,z,y,x]
    l_idx = lidar.indices        # (Nl, 4)
    Nr = r_idx.size(0)
    C_lidar = lidar.features.size(1)

    topk = gt_topk
    matched_mask = torch.zeros(Nr, dtype=torch.bool, device=device)
    gt_offsets = torch.zeros((Nr, topk, 3), dtype=radar.features.dtype, device=device)
    gt_feat    = torch.zeros((Nr, topk, C_lidar), dtype=lidar.features.dtype, device=device)
    gt_coords  = torch.zeros((Nr, topk, 4), dtype=l_idx.dtype, device=device)
    candidate_valid = torch.zeros((Nr, topk), dtype=torch.bool, device=device)

    batch_ids = r_idx[:, 0].unique()
    for b in batch_ids:
        r_mask = (r_idx[:, 0] == b)
        l_mask = (l_idx[:, 0] == b)

        r_inds_b = r_mask.nonzero(as_tuple=False).squeeze(1)
        l_inds_b = l_mask.nonzero(as_tuple=False).squeeze(1)

        if l_inds_b.numel() == 0:
            continue

        r_pos = r_idx[r_inds_b, 1:].to(torch.float32)  # (z,y,x)
        l_pos = l_idx[l_inds_b, 1:].to(torch.float32)

        dist = torch.cdist(r_pos, l_pos, p=2)         # (Nb_r, Nb_l)
        k = min(topk, l_inds_b.numel())
        nn_dist, nn_idx = torch.topk(dist, k=k, dim=1, largest=False)  # (Nb_r, k)

        l_rows = l_inds_b[nn_idx]                     # (Nb_r, k)
        valid_b = (nn_dist <= R) if R is not None else torch.ones_like(nn_dist, dtype=torch.bool)

        matched_mask[r_inds_b] = valid_b.any(dim=1)
        gt_feat[r_inds_b, :k] = lidar.features[l_rows]    # (Nb_r, k, C)

        offs = (l_idx[l_rows, 1:] - r_idx[r_inds_b, 1:].unsqueeze(1)).to(radar.features.dtype)  # (Nb_r, k, 3)
        gt_offsets[r_inds_b, :k] = offs
        gt_coords[r_inds_b, :k] = l_idx[l_rows]
        candidate_valid[r_inds_b, :k] = valid_b

    return matched_mask, gt_offsets, gt_feat, gt_coords.reshape(-1, 4), candidate_valid


import torch
import torch.nn as nn
import spconv.pytorch as spconv
from spconv.pytorch import SparseConvTensor, SubMConv3d, SparseConv3d, SparseInverseConv3d

# keep your subm_block(...) unchanged

class SparseUNet3D_MDN(nn.Module):
    """
    Same backbone as your SparseUNet3D, but head outputs a mixture distribution per voxel:
      - mu (K,3), log_sigma (K,3), mix_logits (K,1)
      - optional: intensity mean (K,1), occupancy logit (K,1)
    """
    def __init__(self, in_ch=20, base_ch=32, K=5, t_max=None):
        super().__init__()
        C = base_ch
        self.K = K
        self.t_max = t_max.unsqueeze(0).unsqueeze(0) if t_max is not None else None

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

        self.bottleneck = subm_block(C*8, C*8, indice_key="subm3")

        # ---------------- Decoder ----------------
        self.up2 = SparseInverseConv3d(C*8, C*4, kernel_size=2, bias=False, indice_key="down2")
        self.dec2 = subm_block(C*8, C*4, indice_key="subm2")

        self.up1 = SparseInverseConv3d(C*4, C*2, kernel_size=2, bias=False, indice_key="down1")
        self.dec1 = subm_block(C*4, C*2, indice_key="subm1")

        self.up0 = SparseInverseConv3d(C*2, C, kernel_size=2, bias=False, indice_key="down0")
        self.dec0 = subm_block(C*2, C, indice_key="subm0")

        # Head: per slot: mu(3) + log_sigma(3) + mu_int(1) + occ_logit(1) + mix_logit(1) = 9
        self.pred_head = spconv.SparseSequential(
            SubMConv3d(base_ch, K*9, kernel_size=1, bias=True, indice_key="head_feat")
        )

    @staticmethod
    def _cat_if_same_coords(a: SparseConvTensor, b: SparseConvTensor):
        if not (a.indices.shape == b.indices.shape and torch.equal(a.indices, b.indices)):
            raise RuntimeError("Skip connection coords mismatch; check indice_key/stride path.")
        feats = torch.cat([a.features, b.features], dim=1)
        return a.replace_feature(feats)

    def forward(self, x: SparseConvTensor):
        e0 = self.enc0(x)
        d0 = self.down0(e0).replace_feature(self.act0(self.bn0(self.down0(e0).features)))

        e1 = self.enc1(d0)
        d1 = self.down1(e1).replace_feature(self.act1(self.bn1(self.down1(e1).features)))

        e2 = self.enc2(d1)
        d2 = self.down2(e2).replace_feature(self.act2(self.bn2(self.down2(e2).features)))

        b  = self.bottleneck(d2)

        u2 = self.up2(b)
        c2 = self._cat_if_same_coords(u2, e2)
        c2 = self.dec2(c2)

        u1 = self.up1(c2)
        c1 = self._cat_if_same_coords(u1, e1)
        c1 = self.dec1(c1)

        u0 = self.up0(c1)
        c0 = self._cat_if_same_coords(u0, e0)
        c0 = self.dec0(c0)

        pred = self.pred_head(c0)          # SparseConvTensor
        N = pred.features.size(0)
        K = self.K

        feats = pred.features.view(N, K, 9)

        mu_off     = feats[:, :, 0:3]                 # (N,K,3) offsets in voxel units
        log_sig_off= feats[:, :, 3:6].clamp(-5, 3)    # (N,K,3) stabilize
        mu_int     = feats[:, :, 6:7]                 # (N,K,1)
        occ_logit  = feats[:, :, 7:8]                 # (N,K,1)
        mix_logit  = feats[:, :, 8:9]                 # (N,K,1)

        return {
            "st": pred,
            "mu_off": mu_off,
            "log_sig_off": log_sig_off,
            "mu_int": mu_int,
            "occ_logit": occ_logit,
            "mix_logit": mix_logit,
        }

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def _std_normal_cdf(x):
    """Standard normal CDF Phi(x), used in Eq. (8)."""
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class SynthLocalLoss_MDN(nn.Module):
    def __init__(self, w_occ=0.2, w_mdn=1.0, w_int=0.1, w_stab=0.2, gt_topk=10, t_max=None, t_min=None, voxel_size=None, origin=None, R=None):
        super().__init__()
        self.w_occ = w_occ
        self.w_mdn = w_mdn
        self.w_int = w_int
        self.w_stab = w_stab  # lambda in Eq. (17), weight on L_stab
        self.gt_topk = gt_topk
        self.R = R  # search radius (voxel units, Sec. IV-B); None disables radius filtering
        self.bce = nn.BCEWithLogitsLoss()
        d = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.t_max = t_max.unsqueeze(0).unsqueeze(0).to(d) if t_max is not None else None
        self.t_min = t_min.unsqueeze(0).unsqueeze(0).to(d) if t_min else -self.t_max if t_max is not None else None
        # origin (meters, xyz) and voxel_size (meters, xyz) are needed to recover the
        # continuous fractional position rho of a matched LiDAR point within its voxel
        # cell (Sec. III-B/C); without them the stability loss cannot be computed.
        self.voxel_size = voxel_size.to(d) if voxel_size is not None else None
        self.origin = origin.to(d) if origin is not None else None

    def voxel_stability_loss(self, mu_m, log_sig_m, mix_logit_m, gt_offsets_m, gt_feat_m, candidate_mask=None):
        """
        Stability-aware loss from Eq. (5)-(9).

        For each matched (radar voxel n, candidate LiDAR point t), we check whether the
        predicted continuous offset keeps the same voxel assignment as the candidate's
        true continuous position, under the model's predicted offset-error distribution
        (the Gaussian mixture emitted by the MDN head).

        mu_m, log_sig_m: (M,K,3) predicted mixture means / log-std of the offset, voxel units, xyz
        mix_logit_m:     (M,K,1) or (M,K) mixture logits
        gt_offsets_m:    (M,T,3) integer part of o* (radar voxel index -> matched LiDAR voxel index), xyz
        gt_feat_m:       (M,T,C) raw matched LiDAR point features (C = P*4, each point is [x,y,z,intensity] in meters)
        candidate_mask:  (M,T) bool, True where the candidate is a real within-R match;
                          invalid (padding / beyond-R) candidates are excluded from the mean.

        Returns: scalar L_stab = E[1 - P_stab] (Eq. 9)
        """
        if self.origin is None or self.voxel_size is None:
            return mu_m.sum() * 0.0

        M, T, _ = gt_offsets_m.shape
        C = gt_feat_m.shape[-1]
        if C % 4 != 0:
            return mu_m.sum() * 0.0

        # ---- rho_d: fractional position of the true aligned point u* within its voxel cell ----
        pts = gt_feat_m.view(M, T, C // 4, 4)                      # (M,T,P,4) xyz(m) + intensity
        valid = (pts.abs().sum(dim=-1) > 0).float()                # padded point slots are all-zero
        n_valid = valid.sum(dim=-1).clamp_min(1.0)                 # (M,T)
        xyz_mean = (pts[..., :3] * valid.unsqueeze(-1)).sum(dim=2) / n_valid.unsqueeze(-1)  # (M,T,3) meters

        u_star = (xyz_mean - self.origin.view(1, 1, 3)) / self.voxel_size.view(1, 1, 3)     # (M,T,3) voxel units
        rho = u_star - torch.floor(u_star)                         # (M,T,3), rho in [0,1)^3

        e_lo = -rho                                                 # e^L_d, Eq. (5)
        e_hi = 1.0 - rho                                            # e^H_d, Eq. (5)
        o_star = gt_offsets_m + rho                                 # true continuous offset, Eq. (1)

        # ---- predicted mixture over the offset error e = o_hat - o* (Eq. 8) ----
        sigma = torch.exp(log_sig_m).clamp_min(1e-3)                # (M,K,3)
        if mix_logit_m.dim() == 3:
            mix_logit_m = mix_logit_m.squeeze(-1)                   # (M,K)
        pi = F.softmax(mix_logit_m, dim=1)                          # (M,K)

        # e = o_hat - o*_t is a shift of the predicted offset distribution by -o*_t
        mu_shift = mu_m.unsqueeze(1) - o_star.unsqueeze(2)          # (M,T,K,3)
        sigma_b = sigma.unsqueeze(1)                                # (M,1,K,3) -> broadcast (M,T,K,3)
        e_hi_b = e_hi.unsqueeze(2)                                  # (M,T,1,3)
        e_lo_b = e_lo.unsqueeze(2)                                  # (M,T,1,3)

        z_hi = (e_hi_b - mu_shift) / sigma_b
        z_lo = (e_lo_b - mu_shift) / sigma_b
        # per-axis independence assumption of Eq. (8), one factor per axis d
        axis_prob = (_std_normal_cdf(z_hi) - _std_normal_cdf(z_lo)).clamp(0.0, 1.0)  # (M,T,K,3)
        comp_prob = axis_prob.prod(dim=-1)                          # (M,T,K)

        # mixture average over K components (generalizes Eq. 8 from a single Gaussian to the MDN mixture)
        p_stab = (pi.unsqueeze(1) * comp_prob).sum(dim=-1)          # (M,T), Eq. (7)/(8)
        r_inst = 1.0 - p_stab                                       # Eq. (9)

        if candidate_mask is not None:
            mask = candidate_mask.float()
            return (r_inst * mask).sum() / mask.sum().clamp_min(1.0)
        return r_inst.mean()

    @staticmethod
    def _mdn_log_prob(mu, log_sigma, mix_logit, y):
        """
        mu:        (M,K,3)
        log_sigma: (M,K,3)
        mix_logit: (M,K,1) or (M,K)
        y:         (M,T,3)
        returns: log p(y) per (M,T) via mixture log-sum-exp
        """
        if mix_logit.ndim == 3:
            mix_logit = mix_logit.squeeze(-1)  # (M,K)

        log_pi = F.log_softmax(mix_logit, dim=1)       # (M,K)

        # Expand for broadcasting
        # y: (M,T,1,3), mu: (M,1,K,3)
        y_ = y.unsqueeze(2)
        mu_ = mu.unsqueeze(1)
        log_sigma_ = log_sigma.unsqueeze(1)

        sigma2_ = torch.exp(2.0 * log_sigma_) + 1e-12  # (M,1,K,3)

        log2pi = np.log(2.0 * np.pi)
        logN = -0.5 * (((y_ - mu_) ** 2) / sigma2_ + 2.0 * log_sigma_ + log2pi).sum(dim=-1)  # (M,T,K)

        log_mix = logN + log_pi.unsqueeze(1)          # (M,T,K)
        return log_mix

    def forward(self, out_dict, radar_st: SparseConvTensor, lidar_st: SparseConvTensor):
        """
        out_dict: output of SparseUNet3D_MDN.forward
        radar_st, lidar_st: sparse tensors for matching
        """
        mu_off      = out_dict["mu_off"]        # (N,K,3) voxel units
        log_sig_off = out_dict["log_sig_off"]   # (N,K,3)
        mu_int      = out_dict["mu_int"]        # (N,K,1)
        occ_logit   = out_dict["occ_logit"]     # (N,K,1)
        mix_logit   = out_dict["mix_logit"]     # (N,K,1)

        matched_mask, gt_offsets_zyx, gt_feat, gt_coords, candidate_valid = local_match_closest_mdn(radar_st, lidar_st, self.gt_topk, R=self.R)
        # matched_mask: (N,), gt_offsets: (N,topk,3) in zyx voxel units
        # print(f"gt_offsets_zyx: {gt_offsets_zyx}")

        # Convert GT offsets to xyz if your mu_off is xyz
        gt_offsets_xyz = torch.flip(gt_offsets_zyx, dims=[-1])  # (N,topk,3) zyx->xyz
        # print(f"gt_offsets_xyz: {gt_offsets_xyz[0][0]}")

        # ---------- Occupancy loss ----------
        # We want "at least one slot exists" when matched_mask is True.
        # A simple and stable approximation: take max over K logits as "any"
        occ_any_logit = occ_logit.squeeze(-1).max(dim=1).values  # (N,)
        occ_loss = self.bce(occ_any_logit, matched_mask.float())

        # ---------- Mixture NLL on offsets ----------
        if matched_mask.any():
            mu_m   = mu_off[matched_mask]         # (M,K,3)
            ls_m   = log_sig_off[matched_mask]    # (M,K,3)
            ml_m   = mix_logit[matched_mask]      # (M,K,1)
            y_m    = gt_offsets_xyz[matched_mask] # (M,T,3)
            cand_valid_m = candidate_valid[matched_mask]  # (M,T), True for real within-R candidates

            log_mix = self._mdn_log_prob(mu_m, ls_m, ml_m, y_m)     # (M,T)
            logp = torch.logsumexp(log_mix, dim=-1)       # (M,T)
            y_mean = y_m.mean(1)
            mu_mean = mu_m.mean(1)
            # true mixture NLL (Eq. 12): uses mu, log_sig, and mix_logit per candidate,
            # averaged only over real within-R candidates (Sec. IV-B)
            cand_mask_f = cand_valid_m.float()
            mdn_nll = -(logp * cand_mask_f).sum() / cand_mask_f.sum().clamp_min(1.0)
            
            # print(f"y_m: {y_mean[0].tolist()}, mu_m: {mu_mean[0].tolist()}")
            # print(f"std (xyz): y: {y_mean.std(0).tolist()}; mu: {mu_mean.std(0).tolist()}")
            # print(f"median (xyz): y: {y_mean.median(0).values.tolist()}; mu: {mu_mean.median(0).values.tolist()}")
            # print(f"95th (xyz): y: {torch.quantile(y_mean, 0.95, dim=0).tolist()}; mu: {torch.quantile(mu_mean, 0.95, dim=0).tolist()}")

            # ---------- Optional: intensity loss weighted by responsibilities ----------
            # Your gt_feat is (N,topk,C). In your earlier code C=20=5*4 (x,y,z,i) in meters.
            # We'll extract GT intensity per topk match as mean over the 5 points.
            int_loss = torch.tensor(0.0, device=mu_off.device)
            if self.w_int > 0 and gt_feat.shape[-1] % 4 == 0:
                C = gt_feat.shape[-1]
                gt_pts = gt_feat.view(gt_feat.shape[0], gt_feat.shape[1], C // 4, 4)  # (N,T,P,4)
                gt_int = gt_pts[..., 3].mean(dim=2)                                   # (N,T)
                gt_int_m = gt_int[matched_mask]                                       # (M,T)

                r = torch.softmax(log_mix, dim=-1)                                            # (M,T,K)

                pred_int_m = mu_int[matched_mask].squeeze(-1)          # (M,K)
                pred_int_m = pred_int_m.unsqueeze(1)                   # (M,1,K)
                gt_int_m   = gt_int_m.unsqueeze(-1)                    # (M,T,1)

                int_elem = r * (pred_int_m - gt_int_m).abs()                    # (M,T,K)
                int_mask = cand_valid_m.unsqueeze(-1).float().expand_as(int_elem)  # (M,T,K)
                int_loss = (int_elem * int_mask).sum() / int_mask.sum().clamp_min(1.0)

            # ---------- Stability-aware loss (Eq. 5-9) ----------
            gt_feat_m = gt_feat[matched_mask]                                        # (M,T,C)
            stab_loss = self.voxel_stability_loss(mu_m, ls_m, ml_m, y_m, gt_feat_m, candidate_mask=cand_valid_m)

        else:
            mdn_nll = mu_off.sum() * 0.0
            int_loss = mdn_nll
            stab_loss = mu_off.sum() * 0.0

        error = mu_m.mean(1) - y_m.mean(1)
        # print(f'mu: min: {torch.min(mu_m)}, max: {torch.max(mu_m)}, isnan: {torch.isnan(mu_m).any()}, isinf: {torch.isinf(mu_m).any()}')
        # print(f'Y:  min: {torch.min(y_m)}, max: {torch.max(y_m)}, isnan: {torch.isnan(y_m).any()}, isinf: {torch.isinf(y_m).any()}')
        # # effective but moderate performanc
        # tol_loss_l = torch.relu(self.t_min - error) 
        # tol_loss_h = torch.relu(error - self.t_max)
        # tol_loss = (tol_loss_l + tol_loss_h)**2

        tol_loss = (abs(error) - self.t_max)**2 # effective but moderate performance
        # print(f'min: {torch.min(error)}, max: {torch.max(error)}, isnan: {torch.isnan(error).any()}, isinf: {torch.isinf(error).any()}')
        # tol_loss = torch.sqrt(t) # nan error
        # tol_loss = (self.voxel_size*(abs(error) - self.t_max))**2 # not helpful at all: converting to meter 
        # # print(f'tol_loss_l: {tol_loss_l}, tol_loss_h: {tol_loss_h}')
        # return self.w_occ * occ_loss + self.w_mdn * mdn_nll + self.w_int * int_loss + 0.2*tol_loss.mean()
        return occ_loss, mdn_nll, int_loss, tol_loss.mean(), stab_loss

import torch
import torch.nn.functional as F

@torch.no_grad()
def sample_points_from_mdn(
    pred_st,                     # SparseConvTensor (for indices)
    mu_off, log_sig_off,         # (N,K,3) voxel units (xyz)
    mix_logit,                   # (N,K,1) or (N,K)
    mu_int=None,                 # (N,K,1) optional
    origin=None,                 # (3,) meters
    vsize_xyz=None,              # (3,) meters
    n_points_per_voxel=5,
    prob_thresh=0.0,
    sample_mode="mixture",       # "mixture" or "top1"
    intensity_mode="mean",       # "mean" or "sample" (if you later model int sigma)
    clamp_intensity=(0.0, None), # (min, max or None)
):
    """
    Returns:
        attrs: (N, P, 4)  [x,y,z (meters), intensity]
        voxel_coords: (N, 4) int [b,z,y,x] (same as pred_st.indices)
        voxel_num_points: (N,) long, number of valid points per voxel (<=P)
        chosen_k: (N,) long, sampled component index
        probs_chosen: (N,) float, mixture probability of chosen_k
    """

    device = pred_st.indices.device
    assert origin is not None and vsize_xyz is not None
    origin = origin.to(device)
    vsize_xyz = vsize_xyz.to(device)

    N, K, _ = mu_off.shape
    P = n_points_per_voxel

    # mixture probs
    if mix_logit.ndim == 3:
        mix_logit = mix_logit.squeeze(-1)  # (N,K)
    pi = F.softmax(mix_logit, dim=1)       # (N,K)

    # choose component per voxel
    if sample_mode == "top1":
        chosen_k = pi.argmax(dim=1)  # (N,)
    elif sample_mode == "mixture":
        chosen_k = torch.multinomial(pi, num_samples=1).squeeze(1)
    else:
        raise ValueError(f"Unknown sample_mode: {sample_mode}")

    probs_chosen = pi[torch.arange(N, device=device), chosen_k]  # (N,)

    # gather params for chosen component
    mu = mu_off[torch.arange(N, device=device), chosen_k]                # (N,3)
    sig = torch.exp(log_sig_off[torch.arange(N, device=device), chosen_k]).clamp_min(1e-3)  # (N,3)

    # print(f'chosen: {chosen_k}, mu: {mu}, mu_off: {mu_off}')
    # base voxel centers in meters
    # pred_st.indices: [b,z,y,x]; we need xyz center
    # flip zyx -> xyz from indices[:,1:4]
    voxel_center_xyz = origin + (torch.flip(pred_st.indices[:, 1:4].float(), dims=[1]) + 0.5) * vsize_xyz  # (N,3)

    # sample offsets in voxel units -> meters
    # sampled_off_vox: (N,P,3)
    eps = torch.randn((N, P, 3), device=device, dtype=mu.dtype)
    sampled_off_vox = mu.unsqueeze(1) + sig.unsqueeze(1) * eps/10  # voxel units
    sampled_off_m = sampled_off_vox * vsize_xyz.view(1, 1, 3)   # meters

    xyz = voxel_center_xyz.unsqueeze(1) + sampled_off_m          # (N,P,3)
    # print(f'xyz: {xyz.shape}')

    # intensity per point
    if mu_int is None:
        inten = torch.zeros((N, P, 1), device=device, dtype=mu.dtype)
    else:
        inten_k = mu_int.squeeze(-1)[torch.arange(N, device=device), chosen_k]  # (N,)
        inten = inten_k.view(N, 1, 1).repeat(1, P, 1)                           # (N,P,1)

    # optional clamp
    if clamp_intensity is not None:
        lo, hi = clamp_intensity
        if lo is not None:
            inten = torch.clamp(inten, min=float(lo))
        if hi is not None:
            inten = torch.clamp(inten, max=float(hi))

    # occupancy / keep rule from mixture confidence (optional)
    keep = (probs_chosen >= prob_thresh)  # (N,)

    # voxel_num_points: either P or 0 (you can make this finer later)
    voxel_num_points = torch.where(
        keep, torch.full((N,), P, device=device, dtype=torch.long),
        torch.zeros((N,), device=device, dtype=torch.long)
    )

    # zero out points if not keep
    xyz = torch.where(keep.view(N, 1, 1), xyz, torch.zeros_like(xyz))
    inten = torch.where(keep.view(N, 1, 1), inten, torch.zeros_like(inten))

    attrs = torch.cat([xyz, inten], dim=-1)  # (N,P,4)

    voxel_coords = pred_st.indices.int()     # (N,4)

    return attrs.contiguous(), voxel_coords, voxel_num_points, chosen_k, probs_chosen, mu
