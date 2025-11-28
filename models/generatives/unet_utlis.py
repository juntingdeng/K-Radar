import torch
import torch.nn as nn
import spconv.pytorch as spconv
from spconv.pytorch import SparseConvTensor, SubMConv3d, SparseConv3d, SparseInverseConv3d

def local_match_new(radar: SparseConvTensor, lidar: SparseConvTensor, R=1):
    assert radar.spatial_shape == lidar.spatial_shape
    Z, Y, X = radar.spatial_shape
    
    # Build an offset cude within range [-R, R]. It's a 3d cube of size [2R+1, 2R+1, 2R+1]
    # Filter out the invalid candidates where no valid lidar points found
    rng = torch.arange(-R, R+1, 1)
    zrng, yrng, xrng = torch.meshgrid(rng, rng, rng) # (2R+1) ^ 3
    offs = torch.stack([zrng, yrng, xrng], dim=-1).cuda() # [2R+1, 2R+1, 2R+1, 3]
    b_rdr, z_rdr, y_rdr, x_rdr = radar.indices[:, 0], radar.indices[:, 1], radar.indices[:, 2], radar.indices[:, 3] # N points
    pos = radar.indices[:, 1:] # [N, 3]
    pos = pos[:, None, None, None, :] #[N, 1, 1, 1, 3]

    # candidates: radar points + offsets, candidate points
    candidates = pos + offs # [N, 2R+1, 2R+1, 2R+1, 3]
    candidates = candidates.view(b_rdr.shape[0], -1, 3) # [N, K, 3]
    print(f'candidates shape: {candidates.shape}')
    b_rdr_k = b_rdr.unsqueeze(dim=1).repeat(1, 27).unsqueeze(dim=-1)
    print(f'b_rdr_k shape: {b_rdr_k.shape}')
    candidates = torch.concat([candidates, b_rdr_k], dim=-1)

    # candidates = sorted(candidates, key=lambda item: _hashmap(item[:, -1], item[:, 0], item[:, 1], item[:, 2], Z, Y, X))
    candidates_sorted = torch.zeros_like(candidates)
    for i in range(b_rdr.shape[0]):
        # print(f'candidates[i]: {candidates[i].shape}')
        candidates[i] = torch.vstack(sorted(candidates[i], key=lambda item: _hashmap(item[-1], item[0], item[1], item[2], Z, Y, X)))
        # print(f'candidates_sorted[i]: {tmp.shape}')
    print(f'candidates_sorted shape: {candidates_sorted.shape}')
    candidates = candidates.view(-1, 4)
    
    b_ldr, z_ldr, y_ldr, x_ldr = lidar.indices[:, 0], lidar.indices[:, 1], lidar.indices[:, 2], lidar.indices[:, 3]
    hashmap_ldr = _hashmap(b_ldr, z_ldr, y_ldr, x_ldr, Z, Y, X)
    print(f'hashmap_ldr: {hashmap_ldr.shape}')
    haspmap_candi = _hashmap(b_rdr_k.reshape(-1), candidates[:, 0], candidates[:, 1], candidates[:, 2], Z, Y, X)
    valid_candi = torch.zeros_like(haspmap_candi, dtype=torch.bool)
    for i in range(haspmap_candi.shape[0]):
        if haspmap_candi[i] in hashmap_ldr:
            haspmap_candi[i] = True
        else:
            haspmap_candi[i] = False
    valid_candi = valid_candi.view(b_rdr.shape[0], -1).unsqueeze(dim=2).repeat(1, 1, 3)
    print(f'valid_cansi: {valid_candi.shape}')

    candidates = candidates.view(b_rdr.shape[0], -1, 4)[:, :, :3] # [N, K, 3]
    candidates_valid = torch.where(valid_candi, candidates, valid_candi).cuda() # [N, K, 3]
    matched = torch.zeros((b_rdr.shape[0], 3)).cuda() # [N, ], one matched lidar position for each rdr point
    for pi in range(candidates_valid.shape[0]):
        for ki in range(candidates_valid.shape[1]):
            if (candidates_valid[ki]).any():
                matched[pi] = candidates_valid[ki]
                break

    matched_indices = matched #_hashmap_ins(Z, Y, X, matched)
    gt_offsets = matched_indices - radar.indices[:, :3]
    gt_feat = torch.zeros_like(radar.features)

    for ri in range(b_rdr.shape[0]):
        print(f'ri:{ri}')
        for li in range(b_ldr.shape[0]):
            if (lidar.indices[li][1: ] == matched_indices[ri]).all():
                print('found')
                gt_feat[ri] = lidar.features[li]
                break

    return matched, gt_offsets, gt_feat